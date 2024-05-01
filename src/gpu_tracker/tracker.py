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


class Tracker:
    """
    Runs a sub-process that tracks computational resources of the calling process. Including the compute time, maximum RAM, and maximum GPU RAM usage within a context manager or explicit ``start()`` and ``stop()`` methods.
    Calculated quantities are scaled depending on the units chosen for them (e.g. megabytes vs. gigabytes, hours vs. days, etc.).

    :ivar MaxRAM max_ram: Description of the maximum RAM usage of the process, any descendents it may have, and the operating system overall.
    :ivar MaxGPURAM max_gpu_ram: Description of the maximum GPU RAM usage of the process and any descendents it may have.
    :ivar ComputeTime compute_time: Description of the real compute time i.e. the duration of tracking.
    """
    _USAGE_FILE_TIME_DIFFERENCE = 10.0

    class State(enum.Enum):
        """The state of the Tracker."""
        NEW = 0
        STARTED = 1
        STOPPED = 2

    def __init__(
            self, sleep_time: float = 1.0, ram_unit: str = 'gigabytes', gpu_ram_unit: str = 'gigabytes', time_unit: str = 'hours',
            disable_logs: bool = False, n_join_attempts: int = 5, join_timeout: float = 10.0, process_id: int | None = None):
        """
        :param sleep_time: The number of seconds to sleep in between usage-collection iterations.
        :param ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param gpu_ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param time_unit: One of 'seconds', 'minutes', 'hours', or 'days'.
        :param disable_logs: If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.
        :param n_join_attempts: The number of times the tracker attempts to join its underlying sub-process.
        :param join_timeout: The amount of time the tracker waits for its underlying sub-process to join.
        :param process_id: The ID of the process to track. Defaults to the current process.
        :raises ValueError: Raised if invalid units are provided.
        """
        Tracker._validate_mem_unit(ram_unit)
        Tracker._validate_mem_unit(gpu_ram_unit)
        Tracker._validate_unit(time_unit, valid_units={'seconds', 'minutes', 'hours', 'days'}, unit_type='time')
        self.sleep_time = sleep_time
        self._ram_coefficient: float = {
            'bytes': 1.0,
            'kilobytes': 1 / 1e3,
            'megabytes': 1 / 1e6,
            'gigabytes': 1 / 1e9,
            'terabytes': 1 / 1e12
        }[ram_unit]
        self._gpu_ram_coefficient: float = {
            'bytes': 1e6,
            'kilobytes': 1e3,
            'megabytes': 1.0,
            'gigabytes': 1 / 1e3,
            'terabytes': 1 / 1e6
        }[gpu_ram_unit]
        self._time_coefficient: float = {
            'seconds': 1.0,
            'minutes': 1 / 60,
            'hours': 1 / (60 * 60),
            'days': 1 / (60 * 60 * 24)
        }[time_unit]
        self._stop_event = mproc.Event()
        self._tracking_process = mproc.Process(target=self._track)
        self._core_percent_sums = {key: 0. for key in ['system', 'main', 'descendents', 'combined']}
        self._cpu_percent_sums = {key: 0. for key in ['system', 'main', 'descendents', 'combined']}
        self._tracking_iteration = 1
        self.disable_logs = disable_logs
        self.n_join_attempts = n_join_attempts
        self.join_timeout = join_timeout
        self.process_id = process_id if process_id is not None else os.getpid()
        self._main_process = psutil.Process(self.process_id)
        self._is_linux = platform.system().lower() == 'linux'
        max_ram = MaxRAM(unit=ram_unit, system_capacity=psutil.virtual_memory().total * self._ram_coefficient)
        max_gpu_ram = MaxGPURAM(unit=gpu_ram_unit, system_capacity=self._system_gpu_ram(measurement='total'))
        cpu_utilization = CPUUtilization(system_core_count=psutil.cpu_count())
        compute_time = ComputeTime(unit=time_unit)
        self.resource_usage = ResourceUsage(
            max_ram=max_ram, max_gpu_ram=max_gpu_ram, cpu_utilization=cpu_utilization, compute_time=compute_time)
        self._resource_usage_file = f'.{uuid.uuid1()}.pkl'
        self.state = Tracker.State.NEW

    def __repr__(self):
        return f'State: {self.state.name}'

    def _log_warning(self, warning: str):
        if not self.disable_logs:
            log.warning(warning)

    @staticmethod
    def _validate_mem_unit(unit: str):
        Tracker._validate_unit(unit, valid_units={'bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes'}, unit_type='memory')

    @staticmethod
    def _validate_unit(unit: str, valid_units: set[str], unit_type: str):
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(sorted(valid_units))}')

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
            def get_memory_maps(process: psutil.Process) -> list:
                return process.memory_maps(grouped=False)

            memory_maps_list: list[list] = self._map_processes(processes, map_func=get_memory_maps)
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
            def get_rss(process: psutil.Process) -> int:
                return process.memory_info().rss
            total_rss = sum(self._map_processes(processes, map_func=get_rss))
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
        max_gpu_ram = getattr(self.resource_usage.max_gpu_ram, attr)
        setattr(self.resource_usage.max_gpu_ram, attr, max(max_gpu_ram, curr_gpu_ram))

    def _system_gpu_ram(self, measurement: str) -> float:
        command = f'nvidia-smi --query-gpu=memory.{measurement} --format=csv,noheader'
        output = subp.check_output(command.split(), stderr=subp.STDOUT).decode()
        output = output.strip().split('\n')
        usages = [line.replace('MiB', '').strip() for line in output]
        ram_sum = sum([int(usage) for usage in usages if usage != ''])
        return ram_sum * self._gpu_ram_coefficient

    def _update_cpu_utilization(self, percentages: list[float], attr: str):
        cpu_percentages: CPUPercentages = getattr(self.resource_usage.cpu_utilization, attr)

        def update_percentages(percent: float, percent_type: str, percent_sums: dict[str, float]):
            percent_sums[attr] += percent
            mean_percent = percent_sums[attr] / self._tracking_iteration
            setattr(cpu_percentages, f'mean_{percent_type}_percent', mean_percent)
            max_percent: float = getattr(cpu_percentages, f'max_{percent_type}_percent')
            setattr(cpu_percentages, f'max_{percent_type}_percent', max(max_percent, percent))
        core_percent = sum(percentages)
        cpu_percent = core_percent / self.resource_usage.cpu_utilization.system_core_count
        update_percentages(percent=core_percent, percent_type='core', percent_sums=self._core_percent_sums)
        update_percentages(percent=cpu_percent, percent_type='cpu', percent_sums=self._cpu_percent_sums)

    def _update_cpu_utilization_by_process(self, processes: list[psutil.Process], attr: str):
        def get_cpu_percent(process: psutil.Process) -> float:
            return process.cpu_percent()
        percentages = self._map_processes(processes, map_func=get_cpu_percent)
        self._update_cpu_utilization(percentages, attr)

    def _update_n_threads(self, processes: list[psutil.Process], attr: str):
        def get_n_threads(process: psutil.Process):
            return process.num_threads()
        n_threads_list = self._map_processes(processes, get_n_threads)
        n_threads = sum(n_threads_list)
        attr = f'{attr}_n_threads'
        max_n_threads = getattr(self.resource_usage.cpu_utilization, attr)
        setattr(self.resource_usage.cpu_utilization, attr, max(n_threads, max_n_threads))

    def _get_descendents(self) -> list[psutil.Process]:
        return [process for process in self._main_process.children(recursive=True) if process.pid != self._tracking_process.pid]

    def _track(self):
        """
        Continuously tracks computational resource usage until the end of tracking is triggered, either by exiting the context manager or by a call to stop()
        """
        start_time = _testable_time()
        # Simulate a do-while loop so that the tracking is executed at least once.
        while True:
            try:
                # Get the maximum RAM usage.
                self._update_ram(rss_values=self.resource_usage.max_ram.main, processes=[self._main_process])
                self._update_ram(rss_values=self.resource_usage.max_ram.descendents, processes=self._get_descendents())
                # Call children() each time it's used to get an updated list in case the children changed since the above call.
                self._update_ram(rss_values=self.resource_usage.max_ram.combined, processes=[self._main_process] + self._get_descendents())
                self.resource_usage.max_ram.system = max(
                    self.resource_usage.max_ram.system, psutil.virtual_memory().used * self._ram_coefficient)
                # Get the maximum GPU RAM usage.
                memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
                nvidia_smi_output = subp.check_output(memory_used_command.split(), stderr=subp.STDOUT).decode()
                if nvidia_smi_output:
                    process_ids = {self.process_id}
                    self._update_gpu_ram(attr='main', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                    process_ids = {process.pid for process in self._get_descendents()}
                    self._update_gpu_ram(attr='descendents', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                    process_ids.add(self.process_id)
                    self._update_gpu_ram(attr='combined', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                self.resource_usage.max_gpu_ram.system = max(
                    self.resource_usage.max_gpu_ram.system, self._system_gpu_ram(measurement='used'))
                # Get the mean and maximum CPU usages.
                # noinspection PyTypeChecker
                system_core_percentages: list[float] = psutil.cpu_percent(percpu=True)
                self._update_cpu_utilization(percentages=system_core_percentages, attr='system')
                self._update_cpu_utilization_by_process(processes=[self._main_process], attr='main')
                self._update_cpu_utilization_by_process(processes=self._get_descendents(), attr='descendents')
                self._update_cpu_utilization_by_process(
                    processes=[self._main_process] + self._get_descendents(), attr='combined')
                self._update_n_threads(processes=[self._main_process], attr='main')
                self._update_n_threads(processes=self._get_descendents(), attr='descendents')
                self._update_n_threads(processes=[self._main_process] + self._get_descendents(), attr='combined')
                # Update compute time
                self.resource_usage.compute_time.time = (_testable_time() - start_time) * self._time_coefficient
                self._tracking_iteration += 1
                with open(self._resource_usage_file, 'wb') as file:
                    pkl.dump(self.resource_usage, file)
                if self._stop_event.is_set():
                    # Tracking has completed.
                    break
                _testable_sleep(self.sleep_time)
            except psutil.NoSuchProcess:
                self._log_warning('Failed to track a process that does not exist. '
                                  'This possibly resulted from the process completing before it could be tracked.')
            except Exception as error:
                self._log_warning('The following uncaught exception occurred in the tracking process:')
                print(error)

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
            time_since_modified = _testable_time() - os.path.getmtime(self._resource_usage_file)
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
            if type(value) == float:
                dictionary[key] = float(f'{value:.3f}')
            elif type(value) == dict:
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


def _testable_sleep(sleep_time: float):
    """ The time.sleep() function causes issues when mocked in tests, so we create this wrapper that can be safely mocked.

    :return: The result of time.sleep()
    """
    return time.sleep(sleep_time)  # pragma: no cover


def _testable_time():
    """ The time.time() function causes issues when mocked in tests, so we create this wrapper that can be safely mocked.

    :return: The result of time.time()
    """
    return time.time()  # pragma: no cover
