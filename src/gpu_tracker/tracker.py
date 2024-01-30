from __future__ import annotations
import time
import multiprocessing as mproc
import os
import psutil
import subprocess as sproc


class Tracker:
    def __init__(
            self, sleep_time: float = 1.0, include_children: bool = True, ram_unit: str = 'gigabyte', gpu_unit: str = 'gigabyte',
            time_unit: str = 'hour'):
        Tracker._validate_mem_unit(ram_unit)
        Tracker._validate_mem_unit(gpu_unit)
        valid_time_units = {'second', 'minute', 'hour', 'day'}
        Tracker._validate_unit(time_unit, valid_time_units, unit_type='time')
        self.sleep_time = sleep_time
        self.include_children = include_children
        self.ram_unit = ram_unit
        self.gpu_unit = gpu_unit
        self.time_unit = time_unit
        self._ram_coefficient = {
            'byte': 1.0,
            'kilobyte': 1 / 1e3,
            'megabyte': 1 / 1e6,
            'gigabyte': 1 / 1e9,
            'terabyte': 1 / 1e12
        }[ram_unit]
        self._gpu_coefficient = {
            'byte': 1e6,
            'kilobyte': 1e3,
            'megabyte': 1.0,
            'gigabyte': 1 / 1e3,
            'terabyte': 1 / 1e6
        }[gpu_unit]
        self._time_coefficient = {
            'second': 1.0,
            'minute': 1 / 60,
            'hour': 1 / (60 * 60),
            'day': 1 / (60 * 60 * 24)
        }[time_unit]
        self.event = mproc.Event()
        self.queue = mproc.Queue()
        self.subprocess = mproc.Process(target=self._profile)
        self.max_ram = None
        self.max_gpu = None
        self.compute_time = None
        self._time1 = time.time()

    @staticmethod
    def _validate_mem_unit(unit: str):
        valid_units = {'byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte'}
        Tracker._validate_unit(unit, valid_units, unit_type='memory')

    @staticmethod
    def _validate_unit(unit: str, valid_units: set[str], unit_type: str):
        if unit not in valid_units:
            raise ValueError(f'"{unit}" is not a valid {unit_type} unit. Valid values are {", ".join(valid_units)}')

    def _profile(self):
        max_ram = 0
        max_gpu = 0
        while True:
            parent_process_id = os.getppid()
            parent_process = psutil.Process(os.getppid())
            # Get the current RAM usage.
            curr_mem_usage = parent_process.memory_info().rss
            process_ids = {parent_process_id}
            if self.include_children:
                child_processes = parent_process.children()
                process_ids.update(process.pid for process in child_processes)
                for child_process in child_processes:
                    child_proc_usage = child_process.memory_info().rss
                    curr_mem_usage += child_proc_usage
            # Get the current GPU memory usage.
            curr_gpu_usage = 0
            memory_used_command = 'nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader'
            nvidia_smi_output = sproc.check_output(memory_used_command.split(), stderr=sproc.STDOUT).decode()
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
            time.sleep(self.sleep_time)
            if self.event.is_set():
                break
        max_ram, max_gpu, compute_time = (
            max_ram * self._ram_coefficient, max_gpu * self._gpu_coefficient, (time.time() - self._time1) * self._time_coefficient)
        self.queue.put((max_ram, max_gpu, compute_time))

    def __enter__(self) -> Tracker:
        self.subprocess.start()
        return self

    def __exit__(self, *_):
        self.event.set()
        self.max_ram, self.max_gpu, self.compute_time = self.queue.get()
        self.subprocess.join()
        self.subprocess.close()
