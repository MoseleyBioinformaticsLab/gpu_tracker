from __future__ import annotations
import time
import threading as thrd
import os
import psutil
import subprocess as subp
import logging as log
import sys

class Tracker:
    def __init__(
            self, sleep_time: float = 1.0, include_children: bool = True, ram_unit: str = 'gigabytes', gpu_unit: str = 'gigabytes',
            time_unit: str = 'hours', n_join_attempts: int = 5, join_timeout: float = 10.0, kill_if_join_fails: bool = False):
        Tracker._validate_mem_unit(ram_unit)
        Tracker._validate_mem_unit(gpu_unit)
        Tracker._validate_unit(time_unit, valid_units={'seconds', 'minutes', 'hours', 'days'}, unit_type='time')
        self.sleep_time = sleep_time
        self.include_children = include_children
        self.ram_unit = ram_unit
        self.gpu_unit = gpu_unit
        self.time_unit = time_unit
        self._ram_coefficient: float = {
            'bytes': 1.0,
            'kilobytes': 1 / 1e3,
            'megabytes': 1 / 1e6,
            'gigabytes': 1 / 1e9,
            'terabytes': 1 / 1e12
        }[ram_unit]
        self._gpu_coefficient: float = {
            'bytes': 1e6,
            'kilobytes': 1e3,
            'megabytes': 1.0,
            'gigabytes': 1 / 1e3,
            'terabytes': 1 / 1e6
        }[gpu_unit]
        self._time_coefficient: float = {
            'seconds': 1.0,
            'minutes': 1 / 60,
            'hours': 1 / (60 * 60),
            'days': 1 / (60 * 60 * 24)
        }[time_unit]
        self.stop_event = thrd.Event()
        self.thread = thrd.Thread(target=self._profile)
        self.max_ram = None
        self.max_gpu = None
        self.compute_time = None
        self.n_join_attempts = n_join_attempts
        self.join_timeout = join_timeout
        self.kill_if_join_fails = kill_if_join_fails

    @staticmethod
    def _validate_mem_unit(unit: str):
        Tracker._validate_unit(unit, valid_units={'bytes', 'kilobytes', 'megabytes', 'gigabytes', 'terabytes'}, unit_type='memory')

    @staticmethod
    def _validate_unit(unit: str, valid_units: set[str], unit_type: str):
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(sorted(valid_units))}')

    def _profile(self):
        max_ram = 0
        max_gpu = 0
        start_time = time.time()
        while not self.stop_event.is_set():
            process_id = os.getpid()
            process = psutil.Process(process_id)
            # Get the current RAM usage.
            curr_mem_usage = process.memory_info().rss
            process_ids = {process_id}
            if self.include_children:
                child_processes = process.children()
                process_ids.update(process.pid for process in child_processes)
                for child_process in child_processes:
                    child_proc_usage = child_process.memory_info().rss
                    curr_mem_usage += child_proc_usage
            # Get the current GPU memory usage.
            curr_gpu_usage = 0
            memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
            nvidia_smi_output = subp.check_output(memory_used_command.split(), stderr=subp.STDOUT).decode()
            if nvidia_smi_output:
                nvidia_smi_output = nvidia_smi_output.strip().split('\n')
                for process_info in nvidia_smi_output:
                    pid, megabytes_used = process_info.strip().split(',')
                    pid = int(pid.strip())
                    if pid in process_ids:
                        megabytes_used = int(megabytes_used.replace('MiB', '').strip())
                        curr_gpu_usage += megabytes_used
            # Update maximum resource usage.
            if curr_mem_usage > max_ram:
                max_ram = curr_mem_usage
            if curr_gpu_usage > max_gpu:
                max_gpu = curr_gpu_usage
            _testable_sleep(self.sleep_time)
            self.max_ram, self.max_gpu, self.compute_time = (
                max_ram * self._ram_coefficient, max_gpu * self._gpu_coefficient, (time.time() - start_time) * self._time_coefficient)

    def __enter__(self) -> Tracker:
        self.thread.start()
        return self

    def __exit__(self, *_):
        n_join_attempts = 0
        while n_join_attempts < self.n_join_attempts:
            self.stop_event.set()
            self.thread.join(timeout=self.join_timeout)
            n_join_attempts += 1
            if self.thread.is_alive():
                log.warning('Thread is still alive after join timout. Attempting to join again...')
            else:
                break
        if self.thread.is_alive():
            log.warning(
                f'Thread is still alive after {self.n_join_attempts} attempts to join. '
                f'The thread will likely not end until the parent process ends.')
            if self.kill_if_join_fails:
                log.warning('The thread failed to join and kill_if_join_fails is set. Exiting ...')
                sys.exit(1)

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__()

    def __str__(self):
        return f'Max RAM: {self.max_ram:.3f} {self.ram_unit}\n' \
               f'Max GPU: {self.max_gpu:.3f} {self.gpu_unit}\n' \
               f'Compute time: {self.compute_time:.3f} {self.time_unit}'

    def __repr__(self):
        return str(self)  # pragma: no cover


def _testable_sleep(sleep_time: float) -> float:
    """ The time.sleep() function causes issues when mocked in tests, so we create this wrapper that can be safely mocked.

    :return: The result of time.sleep()
    """
    return time.sleep(sleep_time)  # pragma: no cover
