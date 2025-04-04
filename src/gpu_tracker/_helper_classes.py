from __future__ import annotations
import abc
import subprocess as subp
import pandas as pd
import io
import os
import csv
import dataclasses as dclass
import sqlalchemy as sqlalc
import sqlalchemy.orm as sqlorm
import enum


class _GPUQuerier(abc.ABC):
    command = None

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
    def _query_gpu(cls, *args: str, ram_column: str):
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
    def _query_gpu(cls, *args: str, ram_column: str) -> pd.DataFrame:
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


@dclass.dataclass
class _TimepointUsage:
    main_ram: float = 0.0
    descendants_ram: float = 0.0
    combined_ram: float = 0.0
    system_ram: float = 0.0
    main_gpu_ram: float = 0.0
    descendants_gpu_ram: float = 0.0
    combined_gpu_ram: float = 0.0
    system_gpu_ram: float = 0.0
    gpu_sum_utilization_percent: float = 0.0
    gpu_hardware_utilization_percent: float = 0.0
    main_n_threads: int = 0
    descendants_n_threads: int = 0
    combined_n_threads: int = 0
    cpu_system_sum_utilization_percent: float = 0.0,
    cpu_system_hardware_utilization_percent: float = 0.0
    cpu_main_sum_utilization_percent: float = 0.0
    cpu_main_hardware_utilization_percent: float = 0.0
    cpu_descendants_sum_utilization_percent: float = 0.0
    cpu_descendants_hardware_utilization_percent: float = 0.0
    cpu_combined_sum_utilization_percent: float = 0.0
    cpu_combined_hardware_utilization_percent: float = 0.0
    timestamp: float = 0.0


@dclass.dataclass
class _SubTrackerLog:
    class CodeBlockPosition(enum.Enum):
        START = 'START'
        STOP = 'STOP'
    process_id: int
    code_block_name: str
    position: CodeBlockPosition
    timestamp: float


class _Writer(abc.ABC):
    @staticmethod
    def create(file: str | None) -> _Writer | None:
        if file is not None:
            if file.endswith('.csv'):
                return _CSVWriter(file)
            elif file.endswith('.sqlite'):
                return _SQLiteWriter(file)
            else:
                raise ValueError(
                    f'Invalid file name: "{file}". Valid file extensions are ".csv" and ".sqlite".')
        else:
            return None

    def __init__(self, file: str):
        self._file = file

    def write_row(self, values: object):
        values = dclass.asdict(values)
        if not os.path.isfile(self._file):
            self._create_file(values)
        self._write_row(values)

    @abc.abstractmethod
    def _write_row(self, values: dict):
        pass  # pragma: nocover

    @abc.abstractmethod
    def _create_file(self, values: dict):
        pass  # pragma: nocover


class _CSVWriter(_Writer):
    def _write_row(self, values: dict):
        with open(self._file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=values.keys())
            writer.writerow(values)

    def _create_file(self, values: dict):
        with open(self._file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=values.keys())
            writer.writeheader()


class _SQLiteWriter(_Writer):
    _DATA_TABLE = 'data'
    _STATIC_DATA_TABLE = 'static_data'

    def _write_row(self, values: dict):
        engine = sqlalc.create_engine(f'sqlite:///{self._file}', poolclass=sqlalc.pool.NullPool)
        metadata = sqlalc.MetaData()
        tracking_table = sqlalc.Table(_SQLiteWriter._DATA_TABLE, metadata, autoload_with=engine)
        Session = sqlorm.sessionmaker(bind=engine)
        with Session() as session:
            insert_stmt = sqlalc.insert(tracking_table).values(**values)
            session.execute(insert_stmt)
            session.commit()

    def _create_file(self, values: dict):
        engine = sqlalc.create_engine(f'sqlite:///{self._file}', poolclass=sqlalc.pool.NullPool)
        metadata = sqlalc.MetaData()
        type_mapping = {
            str: sqlalc.String,
            int: sqlalc.Integer,
            float: sqlalc.Float,
        }
        columns = list[sqlalc.Column]()
        schema = {name: type(value) for name, value in values.items()}
        for column_name, data_type in schema.items():
            sqlalchemy_type = type_mapping[data_type]
            columns.append(sqlalc.Column(column_name, sqlalchemy_type))
        sqlalc.Table(_SQLiteWriter._DATA_TABLE, metadata, *columns)
        metadata.create_all(engine)
