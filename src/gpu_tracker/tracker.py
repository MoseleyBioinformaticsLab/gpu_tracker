"""The ``tracker`` module contains the ``Tracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
from __future__ import annotations
import json
import dataclasses as dclass
import platform
import time
import multiprocessing as mproc
import os
import typing as typ
import psutil
import subprocess as subp
import logging as log
import enum
import pickle as pkl
import uuid


class _TrackingProcess(mproc.Process):
    _CPU_PERCENT_INTERVAL = 0.1
    _ram_unit2coefficient = {
        'bytes': 1.0,
        'kilobytes': 1e-3,
        'megabytes': 1e-6,
        'gigabytes': 1e-9,
        'terabytes': 1e-12
    }
    _gpu_ram_unit2coefficient = {
        'bytes': 1e6,
        'kilobytes': 1e3,
        'megabytes': 1.0,
        'gigabytes': 1e-3,
        'terabytes': 1e-6
    }
    _time_unit2coefficient = {
        'seconds': 1.0,
        'minutes': 1 / 60,
        'hours': 1 / (60 * 60),
        'days': 1 / (60 * 60 * 24)
    }

    def __init__(
            self, stop_event: mproc.Event, sleep_time: float, ram_unit: str, gpu_ram_unit: str, time_unit: str,
            disable_logs: bool, main_process_id: int, resource_usage_file: str, extraneous_process_ids: set[int]):
        super().__init__()
        self._stop_event = stop_event
        if sleep_time < _TrackingProcess._CPU_PERCENT_INTERVAL:
            raise ValueError(
                f'Sleep time of {sleep_time} is invalid. Must be at least {_TrackingProcess._CPU_PERCENT_INTERVAL} seconds.')
        self._sleep_time = sleep_time
        self._ram_coefficient = _TrackingProcess._validate_unit(
            ram_unit, _TrackingProcess._ram_unit2coefficient, unit_type='RAM')
        self._gpu_ram_coefficient = _TrackingProcess._validate_unit(
            gpu_ram_unit, _TrackingProcess._gpu_ram_unit2coefficient, unit_type='GPU RAM')
        self._time_coefficient = _TrackingProcess._validate_unit(
            time_unit, _TrackingProcess._time_unit2coefficient, unit_type='time')
        self._disable_logs = disable_logs
        self._main_process_id = main_process_id
        self._core_percent_sums = {key: 0. for key in ['system', 'main', 'descendents', 'combined']}
        self._cpu_percent_sums = {key: 0. for key in ['system', 'main', 'descendents', 'combined']}
        self._tracking_iteration = 1
        self._is_linux = platform.system().lower() == 'linux'
        self._nvidia_available = True
        try:
            subp.check_output('nvidia-smi')
        except FileNotFoundError:
            self._nvidia_available = False
            self._log_warning(
                'The nvidia-smi command is not available. Please install the Nvidia drivers to track GPU usage. '
                'Otherwise the Max GPU RAM values will remain 0.0')
        max_ram = MaxRAM(unit=ram_unit, system_capacity=psutil.virtual_memory().total * self._ram_coefficient)
        max_gpu_ram = MaxGPURAM(
            unit=gpu_ram_unit, system_capacity=self._system_gpu_ram(measurement='total') if self._nvidia_available else 0.0)
        cpu_utilization = CPUUtilization(system_core_count=psutil.cpu_count())
        compute_time = ComputeTime(unit=time_unit)
        self._resource_usage = ResourceUsage(
            max_ram=max_ram, max_gpu_ram=max_gpu_ram, cpu_utilization=cpu_utilization, compute_time=compute_time)
        self._resource_usage_file = resource_usage_file
        self._extraneous_process_ids = extraneous_process_ids

    def run(self):
        """
        Continuously tracks computational resource usage until the end of tracking is triggered, either by exiting the context manager or by a call to stop()
        """
        start_time = time.time()
        self._extraneous_process_ids.add(self.pid)
        get_cpu_percent = lambda process: process.cpu_percent()
        try:
            main_process = psutil.Process(self._main_process_id)
        except psutil.NoSuchProcess:
            main_process = None
            self._log_warning(f'The target process of ID {self._main_process_id} ended before tracking could begin.')
            self._stop_event.set()
        # Simulate a do-while loop so that the tracking is executed at least once.
        while True:
            with open(self._resource_usage_file, 'wb') as file:
                pkl.dump(self._resource_usage, file)
            if self._stop_event.is_set():
                # Tracking has completed.
                break
            try:
                descendent_processes = [
                    process for process in main_process.children(recursive=True) if process.pid not in self._extraneous_process_ids]
                combined_processes = [main_process] + descendent_processes
                # The first call to cpu_percent returns a meaningless value of 0.0 and should be ignored.
                # And it's recommended to wait a specified amount of time after the first call to cpu_percent.
                # See https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_percent
                self._map_processes(processes=combined_processes, map_func=get_cpu_percent)
                # Get the maximum RAM usage.
                self._update_ram(rss_values=self._resource_usage.max_ram.main, processes=[main_process])
                self._update_ram(rss_values=self._resource_usage.max_ram.descendents, processes=descendent_processes)
                self._update_ram(rss_values=self._resource_usage.max_ram.combined, processes=combined_processes)
                self._resource_usage.max_ram.system = max(
                    self._resource_usage.max_ram.system, psutil.virtual_memory().used * self._ram_coefficient)
                # Get the maximum GPU RAM usage if available.
                if self._nvidia_available:
                    memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
                    nvidia_smi_output = subp.check_output(memory_used_command.split(), stderr=subp.STDOUT).decode()
                    if nvidia_smi_output:
                        process_ids = {self._main_process_id}
                        self._update_gpu_ram(attr='main', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                        process_ids = {
                            process_id for process_id in self._map_processes(
                                processes=descendent_processes, map_func=lambda process: process.pid)}
                        self._update_gpu_ram(attr='descendents', process_ids=process_ids,
                                             nvidia_smi_output=nvidia_smi_output)
                        process_ids.add(self._main_process_id)
                        self._update_gpu_ram(attr='combined', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                    self._resource_usage.max_gpu_ram.system = max(
                        self._resource_usage.max_gpu_ram.system, self._system_gpu_ram(measurement='used'))
                # Get the mean and maximum CPU usages.
                self._update_n_threads(processes=[main_process], attr='main')
                self._update_n_threads(processes=descendent_processes, attr='descendents')
                self._update_n_threads(processes=combined_processes, attr='combined')
                # noinspection PyTypeChecker
                system_core_percentages: list[float] = psutil.cpu_percent(percpu=True)
                self._update_cpu_utilization(percentages=system_core_percentages, attr='system')
                time.sleep(_TrackingProcess._CPU_PERCENT_INTERVAL)
                main_percentage = main_process.cpu_percent()
                descendent_percentages = self._map_processes(processes=descendent_processes, map_func=get_cpu_percent)
                self._update_cpu_utilization(percentages=[main_percentage], attr='main')
                self._update_cpu_utilization(percentages=descendent_percentages, attr='descendents')
                self._update_cpu_utilization(percentages=[main_percentage] + descendent_percentages, attr='combined')
                # Update compute time.
                self._resource_usage.compute_time.time = (time.time() - start_time) * self._time_coefficient
                self._tracking_iteration += 1
                time.sleep(self._sleep_time - _TrackingProcess._CPU_PERCENT_INTERVAL)
            except psutil.NoSuchProcess as error:
                self._log_warning(f'Failed to track a process (PID: {error.pid}) that does not exist. '
                                  f'This possibly resulted from the process completing before it could be tracked.')
            except Exception as error:
                self._log_warning('The following uncaught exception occurred in the tracking process:')
                print(error)

    def _map_processes(self, processes: list[psutil.Process], map_func: typ.Callable[[psutil.Process], typ.Any]) -> list:
        mapped_list = list[list]()
        for process in processes:
            try:
                mapped_list.append(map_func(process))
            except psutil.NoSuchProcess:
                self._log_warning('Attempted to obtain usage information of a process that no longer exists.')
        return mapped_list

    def _update_ram(self, rss_values: RSSValues, processes: list[psutil.Process]):
        if self._is_linux:
            memory_maps_list: list[list] = self._map_processes(
                processes, map_func=lambda process: process.memory_maps(grouped=False))
            private_rss = 0
            path_to_shared_rss = dict[str, float]()
            for memory_maps in memory_maps_list:
                for memory_map in memory_maps:
                    path = memory_map.path
                    # If the same memory map is shared by multiple processes, record the shared rss of the process using the most of it.
                    if path in path_to_shared_rss.keys():
                        path_to_shared_rss[path] = max(path_to_shared_rss[path], memory_map.shared_dirty + memory_map.shared_clean)
                    else:
                        path_to_shared_rss[path] = memory_map.shared_dirty + memory_map.shared_clean
                    private_rss += memory_map.private_dirty + memory_map.private_clean
            private_rss *= self._ram_coefficient
            rss_values.private_rss = max(rss_values.private_rss, private_rss)
            shared_rss = sum(path_to_shared_rss.values()) * self._ram_coefficient
            rss_values.shared_rss = max(rss_values.shared_rss, shared_rss)
            total_rss = private_rss + shared_rss
        else:
            total_rss = sum(self._map_processes(processes, map_func=lambda process: process.memory_info().rss))
            total_rss *= self._ram_coefficient
        rss_values.total_rss = max(rss_values.total_rss, total_rss)

    def _update_gpu_ram(self, attr: str, process_ids: set[int], nvidia_smi_output: str):
        nvidia_smi_output = nvidia_smi_output.strip().split('\n')
        curr_gpu_ram = 0
        for process_info in nvidia_smi_output:
            pid, megabytes_used = process_info.strip().split(',')
            pid = int(pid.strip())
            if pid in process_ids:
                megabytes_used = int(megabytes_used.replace('MiB', '').strip())
                curr_gpu_ram += megabytes_used
        curr_gpu_ram *= self._gpu_ram_coefficient
        max_gpu_ram = getattr(self._resource_usage.max_gpu_ram, attr)
        setattr(self._resource_usage.max_gpu_ram, attr, max(max_gpu_ram, curr_gpu_ram))

    def _system_gpu_ram(self, measurement: str) -> float:
        command = f'nvidia-smi --query-gpu=memory.{measurement} --format=csv,noheader'
        output = subp.check_output(command.split(), stderr=subp.STDOUT).decode()
        output = output.strip().split('\n')
        usages = [line.replace('MiB', '').strip() for line in output]
        ram_sum = sum([int(usage) for usage in usages if usage != ''])
        return ram_sum * self._gpu_ram_coefficient

    def _update_cpu_utilization(self, percentages: list[float], attr: str):
        cpu_percentages: CPUPercentages = getattr(self._resource_usage.cpu_utilization, attr)

        def update_percentages(percent: float, percent_type: str, percent_sums: dict[str, float]):
            percent_sums[attr] += percent
            mean_percent = percent_sums[attr] / self._tracking_iteration
            setattr(cpu_percentages, f'mean_{percent_type}_percent', mean_percent)
            max_percent: float = getattr(cpu_percentages, f'max_{percent_type}_percent')
            setattr(cpu_percentages, f'max_{percent_type}_percent', max(max_percent, percent))

        core_percent = sum(percentages)
        cpu_percent = core_percent / self._resource_usage.cpu_utilization.system_core_count
        update_percentages(percent=core_percent, percent_type='core', percent_sums=self._core_percent_sums)
        update_percentages(percent=cpu_percent, percent_type='cpu', percent_sums=self._cpu_percent_sums)

    def _update_n_threads(self, processes: list[psutil.Process], attr: str):
        n_threads_list = self._map_processes(processes, map_func=lambda process: process.num_threads())
        n_threads = sum(n_threads_list)
        attr = f'{attr}_n_threads'
        max_n_threads = getattr(self._resource_usage.cpu_utilization, attr)
        setattr(self._resource_usage.cpu_utilization, attr, max(n_threads, max_n_threads))

    @staticmethod
    def _validate_unit(unit: str, unit2coefficient: dict[str, float], unit_type: str) -> float:
        valid_units = set(unit2coefficient.keys())
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(sorted(valid_units))}')
        return unit2coefficient[unit]

    def _log_warning(self, warning: str):
        if not self._disable_logs:
            log.warning(warning)


class Tracker:
    """
    Runs a sub-process that tracks computational resources of the calling process. Including the compute time, maximum RAM, and maximum GPU RAM usage within a context manager or explicit ``start()`` and ``stop()`` methods.
    Calculated quantities are scaled depending on the units chosen for them (e.g. megabytes vs. gigabytes, hours vs. days, etc.).

    :ivar resource_usage: Data class containing the max_ram (Description of the maximum RAM usage of the process, any descendents it may have, and the operating system overall), max_gpu_ram (Description of the maximum GPU RAM usage of the process and any descendents it may have), and compute_time (Description of the real compute time i.e. the duration of tracking) attributes.
    """
    _USAGE_FILE_TIME_DIFFERENCE = 10.0

    class State(enum.Enum):
        """The state of the Tracker."""
        NEW = 0
        STARTED = 1
        STOPPED = 2

    def __init__(
            self, sleep_time: float = 1.0, ram_unit: str = 'gigabytes', gpu_ram_unit: str = 'gigabytes', time_unit: str = 'hours',
            disable_logs: bool = False, process_id: int | None = None, n_join_attempts: int = 5, join_timeout: float = 10.0):
        """
        :param sleep_time: The number of seconds to sleep in between usage-collection iterations.
        :param ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param gpu_ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param time_unit: One of 'seconds', 'minutes', 'hours', or 'days'.
        :param disable_logs: If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.
        :param process_id: The ID of the process to track. Defaults to the current process.
        :param n_join_attempts: The number of times the tracker attempts to join its underlying sub-process.
        :param join_timeout: The amount of time the tracker waits for its underlying sub-process to join.
        :raises ValueError: Raised if invalid units are provided.
        """
        current_process_id = os.getpid()
        current_process = psutil.Process(current_process_id)
        # Sometimes a "resource tracker" process is started after creating an Event object.
        # We want to exclude the resource tracker process(es) from the processes being tracked if it's (they're) created.
        # See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        legit_child_ids = {process.pid for process in current_process.children()}
        self._stop_event = mproc.Event()
        extraneous_ids = {process.pid for process in current_process.children()} - legit_child_ids
        self._resource_usage_file = f'.{uuid.uuid1()}.pkl'
        self._tracking_process = _TrackingProcess(
            self._stop_event, sleep_time, ram_unit, gpu_ram_unit, time_unit, disable_logs,
            process_id if process_id is not None else current_process_id, self._resource_usage_file, extraneous_ids)
        self.resource_usage = None
        self.n_join_attempts = n_join_attempts
        self.join_timeout = join_timeout
        self.state = Tracker.State.NEW

    def __repr__(self):
        return f'State: {self.state.name}'

    def __enter__(self) -> Tracker:
        if self.state == Tracker.State.STARTED:
            raise RuntimeError('Cannot start tracking when tracking has already started.')
        elif self.state == Tracker.State.STOPPED:
            raise RuntimeError('Cannot start tracking when tracking has already stopped.')
        self.state = Tracker.State.STARTED
        self._tracking_process.start()
        return self

    def __exit__(self, *_):
        if self.state == Tracker.State.NEW:
            raise RuntimeError('Cannot stop tracking when tracking has not started yet.')
        if self.state == Tracker.State.STOPPED:
            raise RuntimeError('Cannot stop tracking when tracking has already stopped.')
        n_join_attempts = 0
        while n_join_attempts < self.n_join_attempts:
            self._stop_event.set()
            self._tracking_process.join(timeout=self.join_timeout)
            n_join_attempts += 1
            if self._tracking_process.is_alive():
                log.warning('The tracking process is still alive after join timout. Attempting to join again...')
            else:
                break
        if self._tracking_process.is_alive():
            log.warning(
                f'The tracking process is still alive after {self.n_join_attempts} attempts to join. '
                f'Terminating the process by force...')
            self._tracking_process.terminate()
        self._tracking_process.close()
        if os.path.isfile(self._resource_usage_file):
            with open(self._resource_usage_file, 'rb') as file:
                self.resource_usage = pkl.load(file)
            time_since_modified = time.time() - os.path.getmtime(self._resource_usage_file)
            if time_since_modified > Tracker._USAGE_FILE_TIME_DIFFERENCE:
                log.warning(
                    f'Tracking is stopping and it has been {time_since_modified} seconds since the temporary tracking results file was '
                    f'last updated. Resource usage was not updated during that time.')
            os.remove(self._resource_usage_file)
        else:
            raise RuntimeError('The temporary tracking results file does not exist. Tracking results cannot be obtained.')
        self.state = Tracker.State.STOPPED

    def start(self):
        """
        Begins tracking for the duration of time until ``stop()`` is called. Equivalent to entering the context manager.
        """
        self.__enter__()

    def stop(self):
        """
        Stop tracking. Equivalent to exiting the context manager.
        """
        self.__exit__()

    def __str__(self) -> str:
        """
        Constructs a string representation of the computational-resource-usage measurements and their units.
        """
        tracker_json = self.to_json()
        Tracker._format_float(dictionary=tracker_json)
        indent = 3
        text = json.dumps(tracker_json, indent=indent)
        # Un-indent the lines to the left after removing curley braces.
        text = '\n'.join(line[indent:] for line in text.split('\n') if line.strip() not in {'{', '}', '},'})
        text = text.replace(': {', ':').replace('{', '').replace('}', '').replace('_', ' ').replace('"', '').replace(',', '')
        return text.replace('max', 'Max').replace('ram', 'RAM').replace('unit', 'Unit').replace('system', 'System').replace(
            'compute', 'Compute').replace('time: ', 'Time: ').replace('rss', 'RSS').replace('total', 'Total').replace(
            'private', 'Private').replace('shared', 'Shared').replace('main', 'Main').replace('descendents', 'Descendents').replace(
            'combined', 'Combined').replace('gpu', 'GPU').replace('mean', 'Mean').replace('cpu', 'CPU').replace(
            'n threads', 'number of threads')

    @staticmethod
    def _format_float(dictionary: dict):
        """
        Recursively formats the floating points in a dictionary to 3 decimal places.
        :param dictionary: The dictionary to format.
        """
        for key, value in dictionary.items():
            if isinstance(value, float):
                dictionary[key] = float(f'{value:.3f}')
            elif isinstance(value, dict):
                Tracker._format_float(value)

    def to_json(self) -> dict[str, dict]:
        """
        Constructs a dictionary of the computational-resource-usage measurements and their units.
        """
        return dclass.asdict(self.resource_usage)


@dclass.dataclass
class RSSValues:
    """
    :param total_rss: The sum of ``private_rss`` and ``shared_rss``.
    :param private_rss: The RAM usage exclusive to a process.
    :param shared_rss: The RAM usage of a process shared with at least one other process.
    """
    total_rss: float = 0.
    private_rss: float = 0.
    shared_rss: float = 0.


@dclass.dataclass
class MaxRAM:
    """
    :param unit: The unit of measurement for RAM e.g. gigabytes.
    :param system_capacity: A constant value for the RAM capacity of the entire operating system.
    :param system: The RAM usage across the entire operating system.
    :param main: The RAM usage of the main process.
    :param descendents: The summed RAM usage of the descendent processes (i.e. child processes, grandchild processes, etc.).
    :param combined: The summed RAM usage of both the main process and any descendent processes it may have.
    """
    unit: str
    system_capacity: float
    system: float = 0.
    main: RSSValues = dclass.field(default_factory=RSSValues)
    descendents: RSSValues = dclass.field(default_factory=RSSValues)
    combined: RSSValues = dclass.field(default_factory=RSSValues)


@dclass.dataclass
class MaxGPURAM:
    """
    :param unit: The unit of measurement for GPU RAM e.g. gigabytes.
    :param main: The GPU RAM usage of the main process.
    :param descendents: The summed GPU RAM usage of the descendent processes (i.e. child processes, grandchild processes, etc.).
    :param combined: The summed GPU RAM usage of both the main process and any descendent processes it may have.
    """
    unit: str
    system_capacity: float
    system: float = 0.
    main: float = 0.
    descendents: float = 0.
    combined: float = 0.


@dclass.dataclass
class CPUPercentages:
    max_core_percent: float = 0.
    max_cpu_percent: float = 0.
    mean_core_percent: float = 0.
    mean_cpu_percent: float = 0.


@dclass.dataclass
class CPUUtilization:
    system_core_count: int
    system: CPUPercentages = dclass.field(default_factory=CPUPercentages)
    main: CPUPercentages = dclass.field(default_factory=CPUPercentages)
    descendents: CPUPercentages = dclass.field(default_factory=CPUPercentages)
    combined: CPUPercentages = dclass.field(default_factory=CPUPercentages)
    main_n_threads: int = 0
    descendents_n_threads: int = 0
    combined_n_threads: int = 0


@dclass.dataclass
class ComputeTime:
    """
    :param unit: The unit of measurement for compute time e.g. hours.
    :param time: The real compute time.
    """
    unit: str
    time: float = 0.


@dclass.dataclass
class ResourceUsage:
    max_ram: MaxRAM
    max_gpu_ram: MaxGPURAM
    cpu_utilization: CPUUtilization
    compute_time: ComputeTime
