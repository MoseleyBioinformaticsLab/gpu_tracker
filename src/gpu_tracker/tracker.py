"""The ``tracker`` module contains the ``Tracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
from __future__ import annotations
import abc
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
import io
import pandas as pd


class _GPUQuerier(abc.ABC):
    @classmethod
    def _query_gpu(cls, *args) -> pd.DataFrame:
        output = subp.check_output((cls.command,) + args, stderr=subp.STDOUT).decode()
        gpu_info = pd.read_csv(io.StringIO(output))
        return gpu_info.map(lambda value: value.strip() if type(value) is str else value)

    @classmethod
    def is_available(cls) -> bool | None:
        try:
            subp.check_output(cls.command)
            return True
        except subp.CalledProcessError:
            return False
        except FileNotFoundError:
            return None

    @classmethod
    @abc.abstractmethod
    def static_info(cls) -> pd.DataFrame:
        pass  # pragma: nocover

    @classmethod
    @abc.abstractmethod
    def process_ram(cls) -> pd.DataFrame:
        pass  # pragma: nocover

    @classmethod
    @abc.abstractmethod
    def ram_and_utilization(cls) -> pd.DataFrame:
        pass  # pragma: nocover

class _NvidiaQuerier(_GPUQuerier):
    command = 'nvidia-smi'

    @classmethod
    def _query_gpu(cls, *args: list[str], ram_column: str):
        gpu_info = super()._query_gpu(*args, '--format=csv')
        gpu_info.columns = [col.replace('[MiB]', '').replace('[%]', '').strip() for col in gpu_info.columns]
        gpu_info[ram_column] = gpu_info[ram_column].apply(lambda ram: int(ram.replace('MiB', '').strip()))
        return gpu_info.rename(columns={ram_column: 'ram'})

    @classmethod
    def static_info(cls) -> pd.DataFrame:
        return cls._query_gpu('--query-gpu=uuid,memory.total', ram_column='memory.total')

    @classmethod
    def process_ram(cls) -> pd.DataFrame:
        return cls._query_gpu('--query-compute-apps=pid,used_gpu_memory', ram_column='used_gpu_memory')

    @classmethod
    def ram_and_utilization(cls) -> pd.DataFrame:
        gpu_info = cls._query_gpu('--query-gpu=uuid,memory.used,utilization.gpu', ram_column='memory.used')
        gpu_info = gpu_info.rename(columns={'utilization.gpu': 'utilization_percent'})
        gpu_info.utilization_percent = [float(percentage.replace('%', '').strip()) for percentage in gpu_info.utilization_percent]
        return gpu_info


class _AMDQuerier(_GPUQuerier):
    command = 'amd-smi'
    __id_to_uuid = None

    @classmethod
    @property
    def _id_to_uuid(cls) -> dict[int, str]:
        if cls.__id_to_uuid is None:
            gpu_info = super()._query_gpu('list', '--csv')
            cls.__id_to_uuid = {gpu_id: uuid for gpu_id, uuid in zip(gpu_info.gpu, gpu_info.gpu_uuid)}
        return cls.__id_to_uuid

    @classmethod
    def _query_gpu(cls, *args: list[str], ram_column: str) -> pd.DataFrame:
        gpu_info = super()._query_gpu(*args, '--csv')
        if 'gpu' in gpu_info.columns:
            gpu_info.gpu = [cls._id_to_uuid[gpu_id] for gpu_id in gpu_info.gpu]
            gpu_info = gpu_info.rename(columns={'gpu': 'uuid'})
        return gpu_info.rename(columns={ram_column: 'ram'})

    @classmethod
    def static_info(cls) -> pd.DataFrame:
        gpu_info = cls._query_gpu('static', '--vram', ram_column='size')
        return gpu_info[['uuid', 'ram']]

    @classmethod
    def process_ram(cls) -> pd.DataFrame:
        gpu_info = cls._query_gpu('process', ram_column='vram_mem')
        gpu_info.ram = [ram / 1e6 for ram in gpu_info.ram]  # RAM is in bytes for the process subcommand.
        return gpu_info[['pid', 'ram']]

    @classmethod
    def ram_and_utilization(cls) -> pd.DataFrame:
        gpu_info = cls._query_gpu('monitor', '--vram-usage', '--gfx', ram_column='vram_used')
        gpu_info = gpu_info[['uuid', 'gfx', 'ram']]
        gpu_info.gfx = gpu_info.gfx.astype(float)
        return gpu_info.rename(columns={'gfx': 'utilization_percent'})


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
            n_expected_cores: int | None, gpu_uuids: set[str] | None, disable_logs: bool, main_process_id: int,
            resource_usage_file: str, extraneous_process_ids: set[int], gpu_brand: str | None):
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
        percent_keys = ['cpu_system', 'cpu_main', 'cpu_descendants', 'cpu_combined', 'gpu']
        self._sum_percent_sums = {key: 0. for key in percent_keys}
        self._hardware_percent_sums = {key: 0. for key in percent_keys}
        self._tracking_iteration = 1
        self._is_linux = platform.system().lower() == 'linux'
        cannot_connect_warning = ('The {} command is installed but cannot connect to a GPU. '
                                  'The GPU RAM and GPU utilization values will remain 0.0.')
        if gpu_brand is None:
            nvidia_available = _NvidiaQuerier.is_available()
            nvidia_installed = nvidia_available is not None
            nvidia_available = bool(nvidia_available)
            amd_available = _AMDQuerier.is_available()
            amd_installed = amd_available is not None
            amd_available = bool(amd_available)
            if nvidia_available:
                gpu_brand = 'nvidia'
            elif amd_available:
                gpu_brand = 'amd'
            elif nvidia_installed:
                self._log_warning(cannot_connect_warning.format('nvidia-smi'))
            elif amd_installed:
                self._log_warning(cannot_connect_warning.format('amd-smi'))
            else:
                self._log_warning(
                    'Neither the nvidia-smi command nor the amd-smi command is installed. Install one of these to profile the GPU. '
                    'Otherwise the GPU RAM and GPU utilization values will remain 0.0.')
        if gpu_brand == 'nvidia':
            self._gpu_querier = _NvidiaQuerier
        elif gpu_brand == 'amd':
            self._gpu_querier = _AMDQuerier
        elif gpu_brand is None:
            self._gpu_querier = None
        else:
            raise ValueError(f'"{gpu_brand}" is not a valid GPU brand. Supported values are "nvidia" and "amd".')
        max_ram = MaxRAM(unit=ram_unit, system_capacity=psutil.virtual_memory().total * self._ram_coefficient)
        system_core_count = psutil.cpu_count()
        cpu_utilization = CPUUtilization(
            system_core_count=system_core_count,
            n_expected_cores=n_expected_cores if n_expected_cores is not None else system_core_count)
        if self._gpu_querier:
            gpu_info = self._gpu_querier.static_info()
            gpu_ram_system_capacity = self._get_gpu_ram(gpu_info=gpu_info)
            max_gpu_ram = MaxGPURAM(unit=gpu_ram_unit, system_capacity=gpu_ram_system_capacity)
            all_uuids = set(gpu_info.uuid)
            if gpu_uuids is None:
                self._gpu_uuids = all_uuids
            else:
                if len(gpu_uuids) == 0:
                    raise ValueError('gpu_uuids is not None but the set is empty. Please provide a set of at least one GPU UUID.')
                for gpu_uuid in gpu_uuids:
                    if gpu_uuid not in all_uuids:
                        raise ValueError(f'GPU UUID of {gpu_uuid} is not valid. Available UUIDs are: {", ".join(sorted(all_uuids))}')
                self._gpu_uuids = gpu_uuids
            gpu_utilization = GPUUtilization(system_gpu_count=len(all_uuids), n_expected_gpus=len(self._gpu_uuids))
        else:
            max_gpu_ram = MaxGPURAM(unit=gpu_ram_unit, system_capacity=0.0)
            gpu_utilization = GPUUtilization(system_gpu_count=0, n_expected_gpus=0)
        compute_time = ComputeTime(unit=time_unit)
        self._resource_usage = ResourceUsage(
            max_ram=max_ram, max_gpu_ram=max_gpu_ram, cpu_utilization=cpu_utilization, gpu_utilization=gpu_utilization,
            compute_time=compute_time)
        self._resource_usage_file = resource_usage_file
        self._extraneous_process_ids = extraneous_process_ids

    def run(self):
        """
        Continuously tracks computational resource usage until the end of tracking is triggered, either by exiting the context manager or by a call to stop()
        """
        start_time = time.time()
        self._extraneous_process_ids.add(self.pid)
        get_memory_maps = lambda process: process.memory_maps(grouped=False)
        get_rss = lambda process: process.memory_info().rss
        get_n_threads = lambda process: process.num_threads()
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
                descendant_processes = [
                    process for process in main_process.children(recursive=True) if process.pid not in self._extraneous_process_ids]
                # The first call to cpu_percent returns a meaningless value of 0.0 and should be ignored.
                # And it's recommended to wait a specified amount of time after the first call to cpu_percent.
                # See https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_percent
                self._map_processes(processes=[main_process] + descendant_processes, map_func=get_cpu_percent)
                # Get the maximum RAM usage.
                ram_map_func = get_memory_maps if self._is_linux else get_rss
                main_ram = self._map_processes([main_process], map_func=ram_map_func)
                descendants_ram = self._map_processes(descendant_processes, map_func=ram_map_func)
                combined_ram = main_ram + descendants_ram
                kwarg = 'memory_maps_list' if self._is_linux else 'rss_list'
                self._update_ram(rss_values=self._resource_usage.max_ram.main, **{kwarg: main_ram})
                self._update_ram(rss_values=self._resource_usage.max_ram.descendants, **{kwarg: descendants_ram})
                self._update_ram(rss_values=self._resource_usage.max_ram.combined, **{kwarg: combined_ram})
                self._resource_usage.max_ram.system = max(
                    self._resource_usage.max_ram.system, psutil.virtual_memory().used * self._ram_coefficient)
                # Get the maximum GPU RAM usage if available.
                if self._gpu_querier:  # pragma: nocover
                    gpu_info = self._gpu_querier.process_ram()
                    if len(gpu_info):
                        process_ids = {self._main_process_id}
                        self._update_gpu_ram(attr='main', process_ids=process_ids, gpu_info=gpu_info)
                        process_ids = set(self._map_processes(processes=descendant_processes, map_func=lambda process: process.pid))
                        self._update_gpu_ram(attr='descendants', process_ids=process_ids, gpu_info=gpu_info)
                        process_ids.add(self._main_process_id)
                        self._update_gpu_ram(attr='combined', process_ids=process_ids, gpu_info=gpu_info)
                    gpu_info = self._gpu_querier.ram_and_utilization()
                    system_gpu_ram = self._get_gpu_ram(gpu_info)
                    self._resource_usage.max_gpu_ram.system = max(self._resource_usage.max_gpu_ram.system, system_gpu_ram)
                    gpu_info = gpu_info.loc[[uuid in self._gpu_uuids for uuid in gpu_info.uuid]]
                    self._update_processing_unit_utilization(
                        current_percentages=list(gpu_info.utilization_percent),
                        processing_unit_percentages=self._resource_usage.gpu_utilization.gpu_percentages, percent_key='gpu',
                        n_hardware_units=self._resource_usage.gpu_utilization.n_expected_gpus)
                # Get the mean and maximum CPU usages.
                main_n_threads = self._map_processes([main_process], map_func=get_n_threads)
                descendant_n_threads = self._map_processes(descendant_processes, map_func=get_n_threads)
                self._update_n_threads(n_threads_list=main_n_threads, attr='main')
                self._update_n_threads(n_threads_list=descendant_n_threads, attr='descendants')
                self._update_n_threads(n_threads_list=main_n_threads + descendant_n_threads, attr='combined')
                # noinspection PyTypeChecker
                system_core_percentages: list[float] = psutil.cpu_percent(percpu=True)
                cpu_utilization = self._resource_usage.cpu_utilization
                self._update_processing_unit_utilization(
                    current_percentages=system_core_percentages, processing_unit_percentages=cpu_utilization.system,
                    percent_key='cpu_system', n_hardware_units=cpu_utilization.system_core_count)
                time.sleep(_TrackingProcess._CPU_PERCENT_INTERVAL)
                main_percentage = self._map_processes([main_process], map_func=get_cpu_percent)
                descendant_percentages = self._map_processes(processes=descendant_processes, map_func=get_cpu_percent)
                self._update_processing_unit_utilization(
                    current_percentages=main_percentage, processing_unit_percentages=cpu_utilization.main, percent_key='cpu_main',
                    n_hardware_units=cpu_utilization.n_expected_cores)
                self._update_processing_unit_utilization(
                    current_percentages=descendant_percentages, processing_unit_percentages=cpu_utilization.descendants,
                    percent_key='cpu_descendants', n_hardware_units=cpu_utilization.n_expected_cores)
                self._update_processing_unit_utilization(
                    current_percentages=main_percentage + descendant_percentages, processing_unit_percentages=cpu_utilization.combined,
                    percent_key='cpu_combined', n_hardware_units=cpu_utilization.n_expected_cores)
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
            except psutil.NoSuchProcess:  # pragma: nocover
                self._log_warning('Attempted to obtain usage information of a process that no longer exists.')  # pragma: nocover
        return mapped_list

    def _update_ram(self, rss_values: RSSValues, memory_maps_list: list[list] | None = None, rss_list: list[int] | None = None):
        if memory_maps_list is not None:
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
            total_rss = sum(rss_list)
            total_rss *= self._ram_coefficient
        rss_values.total_rss = max(rss_values.total_rss, total_rss)

    def _update_gpu_ram(self, attr: str, process_ids: set[int], gpu_info: pd.DataFrame):
        gpu_info = gpu_info.loc[[pid in process_ids for pid in gpu_info.pid]]
        gpu_ram = self._get_gpu_ram(gpu_info)
        max_gpu_ram = getattr(self._resource_usage.max_gpu_ram, attr)
        setattr(self._resource_usage.max_gpu_ram, attr, max(max_gpu_ram, gpu_ram))

    def _get_gpu_ram(self, gpu_info: pd.DataFrame) -> float:
        return sum(gpu_info.ram) * self._gpu_ram_coefficient

    def _update_processing_unit_utilization(
            self, current_percentages: list[float], processing_unit_percentages: ProcessingUnitPercentages,
            percent_key: str, n_hardware_units: int):
        sum_percent = sum(current_percentages)
        hardware_percent = sum_percent / n_hardware_units
        for percent, percent_sums, percent_type in (
                (sum_percent, self._sum_percent_sums, 'sum'), (hardware_percent, self._hardware_percent_sums, 'hardware')):
            percent_sums[percent_key] += percent
            mean_percent = percent_sums[percent_key] / self._tracking_iteration
            setattr(processing_unit_percentages, f'mean_{percent_type}_percent', mean_percent)
            max_percent: float = getattr(processing_unit_percentages, f'max_{percent_type}_percent')
            setattr(processing_unit_percentages, f'max_{percent_type}_percent', max(max_percent, percent))

    def _update_n_threads(self, n_threads_list: list[int], attr: str):
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
    Runs a sub-process that tracks computational resources of the calling process. Including the compute time, maximum CPU utilization, mean CPU utilization, maximum RAM, and maximum GPU RAM used within a context manager or explicit calls to ``start()`` and ``stop()`` methods.
    Calculated quantities are scaled depending on the units chosen for them (e.g. megabytes vs. gigabytes, hours vs. days, etc.).

    :ivar ResourceUsage resource_usage: Data class containing the computational resource usage data collected by the tracking process.
    """
    _USAGE_FILE_TIME_DIFFERENCE = 10.0

    class State(enum.Enum):
        """The state of the Tracker."""
        NEW = 0
        STARTED = 1
        STOPPED = 2

    def __init__(
            self, sleep_time: float = 1.0, ram_unit: str = 'gigabytes', gpu_ram_unit: str = 'gigabytes', time_unit: str = 'hours',
            n_expected_cores: int = None, gpu_uuids: set[str] = None, disable_logs: bool = False, process_id: int = None,
            resource_usage_file: str | None = None, n_join_attempts: int = 5, join_timeout: float = 10.0,
            gpu_brand: str | None = None):
        """
        :param sleep_time: The number of seconds to sleep in between usage-collection iterations.
        :param ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param gpu_ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param time_unit: One of 'seconds', 'minutes', 'hours', or 'days'.
        :param n_expected_cores: The number of cores expected to be used during tracking (e.g. number of processes spawned, number of parallelized threads, etc.). Used as the denominator when calculating the hardware percentages of the CPU utilization (except for system-wide CPU utilization which always divides by all the cores in the system). Defaults to all the cores in the system.
        :param gpu_uuids: The UUIDs of the GPUs to track utilization for. The length of this set is used as the denominator when calculating the hardware percentages of the GPU utilization (i.e. n_expected_gpus). Defaults to all the GPUs in the system.
        :param disable_logs: If set, warnings are suppressed during tracking. Otherwise, the Tracker logs warnings as usual.
        :param process_id: The ID of the process to track. Defaults to the current process.
        :param resource_usage_file: The file path to the pickle file containing the ``resource_usage`` attribute. This file is automatically deleted and the ``resource_usage`` attribute is set in memory if the tracking successfully completes. But if the tracking is interrupted, the tracking information will be saved in this file as a backup. Defaults to a randomly generated file name in the current working directory of the format ``.gpu-tracker_<random UUID>.pkl``.
        :param n_join_attempts: The number of times the tracker attempts to join its underlying sub-process.
        :param join_timeout: The amount of time the tracker waits for its underlying sub-process to join.
        :param gpu_brand: The brand of GPU to profile. Valid values are "nvidia" and "amd". Defaults to the brand of GPU detected in the system, checking Nvidia first.
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
        self._resource_usage_file = f'.gpu-tracker_{uuid.uuid1()}.pkl' if resource_usage_file is None else resource_usage_file
        self._tracking_process = _TrackingProcess(
            self._stop_event, sleep_time, ram_unit, gpu_ram_unit, time_unit, n_expected_cores, gpu_uuids, disable_logs,
            process_id if process_id is not None else current_process_id, self._resource_usage_file, extraneous_ids, gpu_brand)
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
            raise RuntimeError('The temporary tracking results file does not exist. Tracking results cannot be obtained.')  # pragma: nocover
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
            'private', 'Private').replace('shared', 'Shared').replace('main', 'Main').replace('descendants', 'Descendants').replace(
            'combined', 'Combined').replace('gpu', 'GPU').replace('mean', 'Mean').replace('cpu', 'CPU').replace(
            'n threads', 'number of threads').replace('n expected', 'Number of expected')

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
    The resident set size (RSS) i.e. memory used by a process or processes.

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
    Information related to RAM including the maximum RAM used over a period of time.

    :param unit: The unit of measurement for RAM e.g. gigabytes.
    :param system_capacity: A constant value for the RAM capacity of the entire operating system.
    :param system: The RAM usage across the entire operating system.
    :param main: The RAM usage of the main process.
    :param descendants: The summed RAM usage of the descendant processes (i.e. child processes, grandchild processes, etc.).
    :param combined: The summed RAM usage of both the main process and any descendant processes it may have.
    """
    unit: str
    system_capacity: float
    system: float = 0.
    main: RSSValues = dclass.field(default_factory=RSSValues)
    descendants: RSSValues = dclass.field(default_factory=RSSValues)
    combined: RSSValues = dclass.field(default_factory=RSSValues)


@dclass.dataclass
class MaxGPURAM:
    """
    Information related to GPU RAM including the maximum GPU RAM used over a period of time.

    :param unit: The unit of measurement for GPU RAM e.g. gigabytes.
    :param system_capacity: A constant value for the GPU RAM capacity of all the GPUs in the system.
    :param system: The GPU RAM usage of all the GPUs in the system.
    :param main: The GPU RAM usage of the main process.
    :param descendants: The summed GPU RAM usage of the descendant processes (i.e. child processes, grandchild processes, etc.).
    :param combined: The summed GPU RAM usage of both the main process and any descendant processes it may have.
    """
    unit: str
    system_capacity: float
    system: float = 0.
    main: float = 0.
    descendants: float = 0.
    combined: float = 0.


@dclass.dataclass
class ProcessingUnitPercentages:
    """
    Utilization percentages of one or more processing units (i.e. GPUs or CPU cores).
    Max refers to the highest value measured over a duration of time.
    Mean refers to the average of the measured values during this time.
    Sum refers to the sum of the percentages of the processing units involved. If there is only one unit in question, this is the percentage of just that unit.
    Hardware refers to this sum divided by the number of units involved. If there is only one unit in question, this is the same as the sum.

    :param max_sum_percent: The maximum sum of utilization percentages of the processing units at any given time.
    :param max_hardware_percent: The maximum utilization percentage of the group of units as a whole (i.e. max_sum_percent divided by the number of units involved).
    :param mean_sum_percent: The mean sum of utilization percentages of the processing units used by the process(es) over time.
    :param mean_hardware_percent: The mean utilization percentage of the group of units as a whole (i.e. mean_sum_percent divided by the number of units involved).
    """
    max_sum_percent: float = 0.
    max_hardware_percent: float = 0.
    mean_sum_percent: float = 0.
    mean_hardware_percent: float = 0.


@dclass.dataclass
class CPUUtilization:
    """
    Information related to CPU usage, including core utilization percentages of the main process and any descendant processes it may have as well as system-wide utilization.
    The system hardware utilization percentages are strictly divided by the total number of cores in the system while that of the main, descendant, and combined processes can be divided by the expected number of cores used in a task.

    :param system_core_count: The number of cores available to the entire operating system.
    :param n_expected_cores: The number of cores expected to be used by the main process and/or any descendant processes it may have.
    :param system: The utilization percentages of all the cores in the entire operating system.
    :param main: The utilization percentages of the cores used by the main process.
    :param descendants: The utilization percentages summed across descendant processes (i.e. child processes, grandchild processes, etc.).
    :param combined: The utilization percentages summed across both the descendant processes and the main process.
    :param main_n_threads: The maximum detected number of threads used by the main process at any time.
    :param descendants_n_threads: The maximum sum of threads used across the descendant processes at any time.
    :param combined_n_threads: The maximum sum of threads used by both the main and descendant processes.
    """
    system_core_count: int
    n_expected_cores: int
    system: ProcessingUnitPercentages = dclass.field(default_factory=ProcessingUnitPercentages)
    main: ProcessingUnitPercentages = dclass.field(default_factory=ProcessingUnitPercentages)
    descendants: ProcessingUnitPercentages = dclass.field(default_factory=ProcessingUnitPercentages)
    combined: ProcessingUnitPercentages = dclass.field(default_factory=ProcessingUnitPercentages)
    main_n_threads: int = 0
    descendants_n_threads: int = 0
    combined_n_threads: int = 0


@dclass.dataclass
class GPUUtilization:
    """
    Utilization percentages of one or more GPUs being tracked.
    Hardware percentages are the summed percentages divided by the number of GPUs being tracked.

    :param system_gpu_count: The number of GPUs in the system.
    :param n_expected_gpus: The number of GPUs to be tracked (e.g. GPUs actually used while there may be other GPUs in the system).
    :param gpu_percentages: The utilization percentages of the GPU(s) being tracked.
    """
    system_gpu_count: int
    n_expected_gpus: int
    gpu_percentages: ProcessingUnitPercentages = dclass.field(default_factory=ProcessingUnitPercentages)


@dclass.dataclass
class ComputeTime:
    """
    The time it takes for a task to complete.

    :param unit: The unit of measurement for compute time e.g. hours.
    :param time: The real compute time.
    """
    unit: str
    time: float = 0.


@dclass.dataclass
class ResourceUsage:
    """
    Contains data for computational resource usage.

    :param max_ram: The maximum RAM used at any point while tracking.
    :param max_gpu_ram: The maximum GPU RAM used at any point while tracking.
    :param cpu_utilization: Core counts, utilization percentages of cores and maximum number of threads used while tracking.
    :param gpu_utilization: GPU counts and utilization percentages of the GPU(s).
    :param compute_time: The real time spent tracking.
    """
    max_ram: MaxRAM
    max_gpu_ram: MaxGPURAM
    cpu_utilization: CPUUtilization
    gpu_utilization: GPUUtilization
    compute_time: ComputeTime
