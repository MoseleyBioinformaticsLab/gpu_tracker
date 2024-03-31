"""The ``tracker`` module contains the ``Tracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
from __future__ import annotations
import json
import dataclasses as dclass
import platform
import time
import threading as thrd
import os
import psutil
import subprocess as subp
import logging as log
import sys


@dclass.dataclass
class RSSValues:
    total_rss: float = 0.
    private_rss: float = 0.
    shared_rss: float = 0.


@dclass.dataclass
class MaxRAM:
    main: RSSValues = dclass.field(default_factory=RSSValues)
    descendents: RSSValues = dclass.field(default_factory=RSSValues)
    combined: RSSValues = dclass.field(default_factory=RSSValues)
    system: float = 0.


@dclass.dataclass
class MaxGPURAM:
    main: float = 0.
    descendents: float = 0.
    combined: float = 0.


class Tracker:
    """
    Runs a thread in the background that tracks the compute time, maximum RAM, and maximum GPU RAM usage within a context manager or explicit ``start()`` and ``stop()`` methods.
    Calculated quantities are scaled depending on the unit chosen for them (e.g. megabytes vs. gigabytes, hours vs. days, etc.).

    :ivar dict measurements: The measured values of the computational-resource usage i.e. maximum RAM, maximum GPU RAM, and compute time.
    """
    NO_PROCESS_WARNING = 'Attempted to obtain RAM information of a process that no longer exists.'

    def __init__(
            self, sleep_time: float = 1.0, ram_unit: str = 'gigabytes', gpu_ram_unit: str = 'gigabytes', time_unit: str = 'hours',
            n_join_attempts: int = 5, join_timeout: float = 10.0, kill_if_join_fails: bool = False, process_id: int | None = None):
        """
        :param sleep_time: The number of seconds to sleep in between usage-collection iterations.
        :param ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param gpu_ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param time_unit: One of 'seconds', 'minutes', 'hours', or 'days'.
        :param n_join_attempts: The number of times the tracker attempts to join its underlying thread.
        :param join_timeout: The amount of time the tracker waits for its underlying thread to join.
        :param kill_if_join_fails: If true, kill the process if the underlying thread fails to join.
        :param process_id: The ID of the process to track. Defaults to the current process.
        :raises ValueError: Raised if invalid units are provided.
        """
        Tracker._validate_mem_unit(ram_unit)
        Tracker._validate_mem_unit(gpu_ram_unit)
        Tracker._validate_unit(time_unit, valid_units={'seconds', 'minutes', 'hours', 'days'}, unit_type='time')
        self.sleep_time = sleep_time
        self.ram_unit = ram_unit
        self.gpu_ram_unit = gpu_ram_unit
        self.time_unit = time_unit
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
        self._stop_event = thrd.Event()
        self._thread = thrd.Thread(target=self._profile)
        self.n_join_attempts = n_join_attempts
        self.join_timeout = join_timeout
        self.kill_if_join_fails = kill_if_join_fails
        self.process_id = process_id if process_id is not None else os.getpid()
        self._main_process = psutil.Process(self.process_id)
        self._is_linux = platform.system().lower() == 'linux'
        self.system_ram_capacity = psutil.virtual_memory().total * self._ram_coefficient
        self.max_ram = MaxRAM()
        self.max_gpu_ram = MaxGPURAM()
        self.compute_time = 0.

    @staticmethod
    def _validate_mem_unit(unit: str):
        Tracker._validate_unit(unit, valid_units={'bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes'}, unit_type='memory')

    @staticmethod
    def _validate_unit(unit: str, valid_units: set[str], unit_type: str):
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(sorted(valid_units))}')

    def _update_ram(self, rss_values: RSSValues, processes: list[psutil.Process]):
        if self._is_linux:
            memory_maps_list = list[list]()
            for process in processes:
                try:
                    memory_maps_list.append(process.memory_maps(grouped=False))
                except psutil.NoSuchProcess:
                    log.warning(self.NO_PROCESS_WARNING)
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
            total_rss = 0
            for process in processes:
                try:
                    total_rss += process.memory_info().rss
                except psutil.NoSuchProcess:
                    log.warning(self.NO_PROCESS_WARNING)
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
        max_gpu_ram = getattr(self.max_gpu_ram, attr)
        setattr(self.max_gpu_ram, attr, max(max_gpu_ram, curr_gpu_ram))

    def _profile(self):
        """
        Continuously tracks computational resource usage until the end of tracking is triggered, either by exiting the context manager or by a call to stop()
        """
        start_time = time.time()
        while not self._stop_event.is_set():
            try:
                # Get the maximum RAM usage.
                self._update_ram(rss_values=self.max_ram.main, processes=[self._main_process])
                self._update_ram(rss_values=self.max_ram.descendents, processes=self._main_process.children(recursive=True))
                # Call children() each time it's used to get an updated list in case the children changed since the above call.
                self._update_ram(
                    rss_values=self.max_ram.combined, processes=[self._main_process] + self._main_process.children(recursive=True))
                self.max_ram.system = max(self.max_ram.system, psutil.virtual_memory().used * self._ram_coefficient)
                # Get the maximum GPU RAM usage.
                memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
                nvidia_smi_output = subp.check_output(memory_used_command.split(), stderr=subp.STDOUT).decode()
                if nvidia_smi_output:
                    process_ids = {self.process_id}
                    self._update_gpu_ram(attr='main', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                    process_ids = {process.pid for process in self._main_process.children(recursive=True)}
                    self._update_gpu_ram(attr='descendents', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                    process_ids.add(self.process_id)
                    self._update_gpu_ram(attr='combined', process_ids=process_ids, nvidia_smi_output=nvidia_smi_output)
                # Update compute time
                self.compute_time = (time.time() - start_time) * self._time_coefficient
                _testable_sleep(self.sleep_time)
            except psutil.NoSuchProcess:
                log.warning('Failed to track a process that does not exist. '
                            'This possibly resulted from the process completing before tracking could begin.')
            except Exception as error:
                log.warning('The following uncaught exception occurred in the Tracker\'s thread:')
                print(error)

    def __enter__(self) -> Tracker:
        self._thread.start()
        return self

    def __exit__(self, *_):
        n_join_attempts = 0
        while n_join_attempts < self.n_join_attempts:
            self._stop_event.set()
            self._thread.join(timeout=self.join_timeout)
            n_join_attempts += 1
            if self._thread.is_alive():
                log.warning('Thread is still alive after join timout. Attempting to join again...')
            else:
                break
        if self._thread.is_alive():
            log.warning(
                f'Thread is still alive after {self.n_join_attempts} attempts to join. '
                f'The thread will likely not end until the parent process ends.')
            if self.kill_if_join_fails:
                log.warning('The thread failed to join and kill_if_join_fails is set. Exiting ...')
                sys.exit(1)

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
        max_ram, max_gpu_ram, compute_time = (
            f'{measurement:.3f} {unit}' if measurement is not None else 'null' for measurement, unit in (
                (self.measurements['max_ram']['combined']['total_rss'], self.ram_unit),
                (self.measurements['max_gpu_ram']['combined'], self.gpu_ram_unit),
                (self.measurements['compute_time'], self.time_unit)))
        return \
            f'Max RAM (combined total RSS): {max_ram}\n' \
            f'Max GPU RAM (combined): {max_gpu_ram}\n' \
            f'Compute time: {compute_time}'

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover

    def to_json(self):
        return {
            'system_ram_capacity': self.system_ram_capacity,
            'max_ram': dclass.asdict(self.max_ram),
            'ram_unit': self.ram_unit,
            'max_gpu_ram': dclass.asdict(self.max_gpu_ram),
            'gpu_ram_unit': self.gpu_ram_unit,
            'compute_time': self.compute_time,
            'time_unit': self.time_unit
        }


def _testable_sleep(sleep_time: float):
    """ The time.sleep() function causes issues when mocked in tests, so we create this wrapper that can be safely mocked.

    :return: The result of time.sleep()
    """
    return time.sleep(sleep_time)  # pragma: no cover
