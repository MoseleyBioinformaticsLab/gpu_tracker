"""The ``tracker`` module contains the ``Tracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
from __future__ import annotations
import time
import threading as thrd
import os
import psutil
import subprocess as subp
import logging as log
import sys


class Tracker:
    """
    Runs a thread in the background that tracks the compute time, maximum RAM, and maximum GPU RAM usage within a context manager or explicit ``start()`` and ``stop()`` methods.
    Calculated quantities are scaled depending on the unit chosen for them (e.g. megabytes vs. gigabytes, hours vs. days, etc.).

    :ivar float max_ram: The highest RAM observed while tracking.
    :ivar float max_gpu_ram: The highest GPU RAM observed while tracking.
    :ivar float compute_time: The amount of time spent tracking.
    """

    def __init__(
            self, sleep_time: float = 1.0, include_children: bool = True, ram_unit: str = 'gigabytes', gpu_ram_unit: str = 'gigabytes',
            time_unit: str = 'hours', n_join_attempts: int = 5, join_timeout: float = 10.0, kill_if_join_fails: bool = False,
            process_id: int | None = None):
        """
        :param sleep_time: The number of seconds to sleep in between usage-collection iterations.
        :param include_children: Whether to add the usage (RAM and GPU RAM) of child processes. Otherwise, only collects usage of the main process.
        :param ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param gpu_ram_unit: One of 'bytes', 'kilobytes', 'megabytes', 'gigabytes', or 'terabytes'.
        :param time_unit: One of 'seconds', 'minutes', 'hours', or 'days'.
        :param n_join_attempts: The number of times the tracker attempts to join its underlying thread.
        :param join_timeout: The amount of time the tracker waits for its underlying thread to join.
        :param kill_if_join_fails: If true, kill the process if the underlying thread fails to join.
        :param process_id: The ID of the process to track (along with its children if ``include_children`` is set). Defaults to the current process.
        :raises ValueError: Raised if invalid units are provided.
        """
        Tracker._validate_mem_unit(ram_unit)
        Tracker._validate_mem_unit(gpu_ram_unit)
        Tracker._validate_unit(time_unit, valid_units={'seconds', 'minutes', 'hours', 'days'}, unit_type='time')
        self.sleep_time = sleep_time
        self.include_children = include_children
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
        self.max_ram = None
        self.max_gpu_ram = None
        self.compute_time = None
        self.n_join_attempts = n_join_attempts
        self.join_timeout = join_timeout
        self.kill_if_join_fails = kill_if_join_fails
        self.process_id = process_id if process_id is not None else os.getpid()

    @staticmethod
    def _validate_mem_unit(unit: str):
        Tracker._validate_unit(unit, valid_units={'bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes'}, unit_type='memory')

    @staticmethod
    def _validate_unit(unit: str, valid_units: set[str], unit_type: str):
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(sorted(valid_units))}')

    def _profile(self):
        """
        Continuously tracks computational resource usage until the end of tracking is triggered, either by exiting the context manager or by a call to stop()
        """
        max_ram = 0
        max_gpu_ram = 0
        start_time = time.time()
        while not self._stop_event.is_set():
            try:
                process = psutil.Process(self.process_id)
                # Get the current RAM usage.
                curr_mem_usage = process.memory_info().rss
                process_ids = {self.process_id}
                if self.include_children:
                    child_processes = process.children()
                    process_ids.update(process.pid for process in child_processes)
                    for child_process in child_processes:
                        try:
                            child_proc_usage = child_process.memory_info().rss
                            curr_mem_usage += child_proc_usage
                        except psutil.NoSuchProcess:
                            log.warning(
                                'Race condition: Failed to collect usage of a previously detected child process that no longer exists.')
                # Get the current GPU RAM usage.
                curr_gpu_ram = 0
                memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
                nvidia_smi_output = subp.check_output(memory_used_command.split(), stderr=subp.STDOUT).decode()
                if nvidia_smi_output:
                    nvidia_smi_output = nvidia_smi_output.strip().split('\n')
                    for process_info in nvidia_smi_output:
                        pid, megabytes_used = process_info.strip().split(',')
                        pid = int(pid.strip())
                        if pid in process_ids:
                            megabytes_used = int(megabytes_used.replace('MiB', '').strip())
                            curr_gpu_ram += megabytes_used
                # Update maximum resource usage.
                if curr_mem_usage > max_ram:
                    max_ram = curr_mem_usage
                if curr_gpu_ram > max_gpu_ram:
                    max_gpu_ram = curr_gpu_ram
                _testable_sleep(self.sleep_time)
                self.max_ram, self.max_gpu_ram, self.compute_time = (
                    max_ram * self._ram_coefficient, max_gpu_ram * self._gpu_ram_coefficient,
                    (time.time() - start_time) * self._time_coefficient)
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

    def to_json(self) -> dict[str, float | str]:
        """
        Constructs a dictionary containing the computational-resource-measurements and their units.
        """
        return {
            'max_ram': self.max_ram,
            'ram_unit': self.ram_unit,
            'max_gpu_ram': self.max_gpu_ram,
            'gpu_ram_unit': self.gpu_ram_unit,
            'compute_time': self.compute_time,
            'time_unit': self.time_unit
        }

    def __str__(self) -> str:
        """
        Constructs a string representation of the computational-resource-usage measurements and their units.
        """
        max_ram, max_gpu_ram, compute_time = (
            f'{measurement:.3f} {unit}' if measurement is not None else 'null' for measurement, unit in (
                (self.max_ram, self.ram_unit), (self.max_gpu_ram, self.gpu_ram_unit), (self.compute_time, self.time_unit)))
        return \
            f'Max RAM: {max_ram}\n' \
            f'Max GPU RAM: {max_gpu_ram}\n' \
            f'Compute time: {compute_time}'

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover


def _testable_sleep(sleep_time: float):
    """ The time.sleep() function causes issues when mocked in tests, so we create this wrapper that can be safely mocked.

    :return: The result of time.sleep()
    """
    return time.sleep(sleep_time)  # pragma: no cover
