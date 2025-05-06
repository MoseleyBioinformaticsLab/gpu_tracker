from __future__ import annotations
import abc
import os.path
import subprocess as subp
import pandas as pd
import io
import csv
import dataclasses as dclass
import sqlalchemy as sqlalc
import sqlalchemy.orm as sqlorm
import enum
import tqdm

_SUMMARY_STATS = ['min', 'max', 'mean', 'std']


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
class _StaticData:
    ram_unit: str
    gpu_ram_unit: str
    time_unit: str
    ram_system_capacity: float
    gpu_ram_system_capacity: float
    system_core_count: int
    n_expected_cores: int
    system_gpu_count: int
    n_expected_gpus: int


@dclass.dataclass
class _SubTrackerLog:
    class CodeBlockPosition(enum.Enum):
        START = 0
        STOP = 1
    process_id: int
    code_block_name: str
    position: CodeBlockPosition
    timestamp: float


class _DataProxy(abc.ABC):
    _files_w_data = set[str]()
    _files_w_static_data = set[str]()

    @staticmethod
    def create(file: str | None, overwrite: bool = False) -> _DataProxy | None:
        if file is not None:
            if file.endswith('.csv'):
                return _CSVDataProxy(file, overwrite)
            elif file.endswith('.sqlite'):
                return _SQLiteDataProxy(file, overwrite)
            else:
                raise ValueError(
                    f'Invalid file name: "{file}". Valid file extensions are ".csv" and ".sqlite".')
        else:
            return None

    def __init__(self, file_name: str, overwrite: bool):
        self._file_name = file_name
        self._overwrite = overwrite
        self._extension = '.csv' if self._file_name.endswith('.csv') else '.sqlite'

    def _check_overwrite(self):
        if os.path.isfile(self._file_name):
            if self._overwrite:
                os.remove(self._file_name)
            else:
                raise FileExistsError(f'File {self._file_name} already exists. Set overwrite to True to overwrite the existing file.')

    def write_data(self, data: _TimepointUsage | _SubTrackerLog):
        data = dclass.asdict(data)
        if self._file_name not in _DataProxy._files_w_data:
            if self._file_name not in _DataProxy._files_w_static_data:
                self._check_overwrite()
            self._create_table(data)
            _DataProxy._files_w_data.add(self._file_name)
        if not os.path.isfile(self._file_name):
            raise FileNotFoundError(f'The file {self._file_name} was removed in the middle of writing data to it.')
        self._write_data(data)

    @abc.abstractmethod
    def _write_data(self, data: dict):
        pass  # pragma: nocover

    @abc.abstractmethod
    def _create_table(self, data: dict):
        pass  # pragma: nocover

    def write_static_data(self, data: _StaticData):
        self._check_overwrite()
        self._write_static_data(data)
        _DataProxy._files_w_static_data.add(self._file_name)

    @abc.abstractmethod
    def _write_static_data(self, data: _StaticData):
        pass  # pragma: nocover

    def read_static_data(self) -> pd.Series:
        return self._read_static_data().squeeze()

    @abc.abstractmethod
    def _read_static_data(self) -> pd.DataFrame:
        pass  # pragma: nocover

    def combine_files(self, files: list[str]):
        if os.path.exists(self._file_name):
            raise ValueError(f'Cannot create sub-tracking file {self._file_name}. File already exists.')
        for file in files:
            if not file.endswith(self._extension):
                raise ValueError(f'File {file} does not end with the same extension as {self._file_name}. Must end in "{self._extension}".')
        self._combine_files(files)

    @abc.abstractmethod
    def _combine_files(self, files: list[str]):
        pass  # pragma: nocover

    def load_timestamp_pairs(self, code_block_name: str) -> list[tuple[float, float]]:
        timestamps = self._load_timestamps(code_block_name)
        indexes_to_drop = list[int]()
        for process_id in timestamps.process_id.unique():
            process_timestamps = timestamps.loc[timestamps.process_id == process_id]
            if process_timestamps.position.iloc[-1] == _SubTrackerLog.CodeBlockPosition.START.value:
                indexes_to_drop.append(process_timestamps.index[-1])
        timestamps = timestamps.drop(indexes_to_drop)
        timestamp_pairs = list[tuple[float, float]]()
        for i in range(0, len(timestamps), 2):
            timestamp1, timestamp2 = timestamps.iloc[i], timestamps.iloc[i + 1]
            start_time, stop_time = float(timestamp1.timestamp), float(timestamp2.timestamp)
            pid1, pid2 = int(timestamp1.process_id), int(timestamp2.process_id)
            error_prefix = f'Sub-tracking file is invalid. Detected timestamp pair ({start_time}, {stop_time})'
            if pid1 != pid2:
                raise ValueError(f'{error_prefix} with differing process IDs: {pid1} and {pid2}.')
            if start_time > stop_time:
                raise ValueError(f'{error_prefix} of process ID {pid1} with a start time greater than the stop time.')
            timestamp_pairs.append((start_time, stop_time))
        return timestamp_pairs

    @abc.abstractmethod
    def _load_timestamps(self, code_block_name: str) -> pd.DataFrame:
        pass  # pragma: nocover

    @abc.abstractmethod
    def load_timepoints(self, timestamp_pairs: list[tuple[float, float]] | None) -> pd.DataFrame:
        pass  # pragma: nocover

    @abc.abstractmethod
    def load_code_block_names(self) -> list[str]:
        pass  # pragma: nocover

    def overall_timepoint_results(self) -> pd.DataFrame:
        fields = list(_TimepointUsage.__dataclass_fields__.keys())
        fields.remove('timestamp')
        return self._overall_timepoint_results(fields)

    @abc.abstractmethod
    def _overall_timepoint_results(self, fields: list[str]) -> pd.DataFrame:
        pass  # pragma: nocover


class _CSVDataProxy(_DataProxy):
    def __init__(self, file_name: str, overwrite: bool):
        super().__init__(file_name, overwrite)
        self._timestamps = None
        self._timepoints = None

    @property
    def timestamps(self):
        if self._timestamps is None:
            self._timestamps = pd.read_csv(self._file_name)
        return self._timestamps

    @property
    def timepoints(self):
        if self._timepoints is None:
            self._timepoints = pd.read_csv(self._file_name, skiprows=2)
        return self._timepoints

    def _write_static_data(self, data: _StaticData):
        if self._file_name in _DataProxy._files_w_data:
            raise RuntimeError('The static data for a CSV file must be created before the dynamic data.')
        static_data = dclass.asdict(data)
        self._create_table(static_data)
        self._write_data(static_data)

    def _write_data(self, data: dict):
        self._with_writer(data, lambda writer: writer.writerow(data))

    def _create_table(self, data: dict):
        self._with_writer(data, lambda writer: writer.writeheader())

    def _with_writer(self, data: dict, func):
        with open(self._file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            func(writer)

    def _read_static_data(self) -> pd.DataFrame:
        return pd.read_csv(self._file_name, header=0, nrows=1)

    def _combine_files(self, files: list[str]):
        data = pd.DataFrame()
        for file in files:
            data = pd.concat([data, pd.read_csv(file)], axis='rows')
        data.to_csv(self._file_name, index=False)

    def _load_timestamps(self, code_block_name: str) -> pd.DataFrame:
        timestamps = self.timestamps.loc[
            self.timestamps.code_block_name == code_block_name, ['process_id', 'position', 'timestamp']
        ]
        return timestamps.sort_values(by=['process_id', 'timestamp', 'position'])

    def load_timepoints(self, timestamp_pairs: list[tuple[float, float]]) -> pd.DataFrame:
        selected = None
        for start_time, stop_time in timestamp_pairs:
            between = (self.timepoints.timestamp >= start_time) & (self.timepoints.timestamp <= stop_time)
            if selected is None:
                selected = between
            else:
                selected |= between
        return self.timepoints[selected]

    def load_code_block_names(self) -> list[str]:
        return sorted(self.timestamps.code_block_name.unique())

    def _overall_timepoint_results(self, fields: list[str]) -> pd.DataFrame:
        return self.timepoints[fields].describe().loc[_SUMMARY_STATS].T


class _SQLiteDataProxy(_DataProxy):
    _DATA_TABLE = 'data'
    _STATIC_DATA_TABLE = 'static_data'

    def _write_data(self, data: dict):
        self.__write_data(data, _SQLiteDataProxy._DATA_TABLE)

    def __write_data(self, data: dict, table: str):
        engine = self._create_engine()
        metadata = sqlalc.MetaData()
        tracking_table = sqlalc.Table(table, metadata, autoload_with=engine)
        Session = sqlorm.sessionmaker(bind=engine)
        with Session() as session:
            insert_stmt = sqlalc.insert(tracking_table).values(**data)
            session.execute(insert_stmt)
            session.commit()

    def _create_table(self, data: dict):
        self.__create_table(data, _SQLiteDataProxy._DATA_TABLE)

    def __create_table(self, data: dict, table: str):
        engine = self._create_engine()
        metadata = sqlalc.MetaData()
        type_mapping = {
            str: sqlalc.String,
            int: sqlalc.Integer,
            float: sqlalc.Float,
        }
        columns = list[sqlalc.Column]()
        schema = {name: type(value) for name, value in data.items()}
        for column_name, data_type in schema.items():
            sqlalchemy_type = type_mapping[data_type]
            columns.append(sqlalc.Column(column_name, sqlalchemy_type))
        sqlalc.Table(table, metadata, *columns)
        metadata.create_all(engine)

    def _create_engine(self) -> sqlalc.Engine:
        return sqlalc.create_engine(f'sqlite:///{self._file_name}', poolclass=sqlalc.pool.NullPool)

    def _read_sql(self, sql: str) -> pd.DataFrame:
        engine = self._create_engine()
        return pd.read_sql(sqlalc.text(sql), engine)

    def _write_static_data(self, data: _StaticData):
        static_data = dclass.asdict(data)
        self.__create_table(static_data, _SQLiteDataProxy._STATIC_DATA_TABLE)
        self.__write_data(static_data, _SQLiteDataProxy._STATIC_DATA_TABLE)

    def _read_static_data(self) -> pd.DataFrame:
        engine = self._create_engine()
        return pd.read_sql_table(_SQLiteDataProxy._STATIC_DATA_TABLE, engine)

    def _combine_files(self, files: list[str]):
        engine = self._create_engine()
        with engine.connect() as con:
            table_created = False
            for in_file in tqdm.tqdm(files):
                con.execute(sqlalc.text(f"ATTACH DATABASE '{in_file}' AS input_db"))
                if not table_created:
                    con.execute(
                        sqlalc.text(
                            f'CREATE TABLE {_SQLiteDataProxy._DATA_TABLE} AS SELECT * FROM input_db.{_SQLiteDataProxy._DATA_TABLE}'
                        )
                    )
                    table_created = True
                else:
                    con.execute(
                        sqlalc.text(
                            f'INSERT INTO {_SQLiteDataProxy._DATA_TABLE} SELECT * FROM input_db.{_SQLiteDataProxy._DATA_TABLE}'
                        )
                    )
                con.commit()
                con.execute(sqlalc.text('DETACH DATABASE input_db'))
                con.commit()

    def _load_timestamps(self, code_block_name: str) -> pd.DataFrame:
        sql = f"""
            SELECT process_id,position,timestamp FROM {_SQLiteDataProxy._DATA_TABLE}
            WHERE code_block_name='{code_block_name}'
            ORDER BY process_id,timestamp,position;
        """
        return self._read_sql(sql)

    def load_timepoints(self, timestamp_pairs: list[tuple[float, float]]) -> pd.DataFrame:
        conditions = [f'timestamp BETWEEN {start_time} AND {stop_time}' for (start_time, stop_time) in timestamp_pairs]
        where_clause = ' OR\n'.join(conditions)
        sql = f'SELECT * FROM {_SQLiteDataProxy._DATA_TABLE} WHERE {where_clause}'
        return self._read_sql(sql)

    def _overall_timepoint_results(self, fields: list[str]) -> pd.DataFrame:
        sql = 'SELECT\n'
        std_func = 'sqrt((sum({0} * {0}) - (sum({0}) * sum({0})) / count({0})) / count({0})) AS "STDDEV({0})"'
        sql_funcs = 'MIN', 'MAX', 'AVG', 'STDDEV'
        field_aggregates = list[str]()
        for func in sql_funcs:
            for field in fields:
                aggregate = f'{func}({field})' if func != 'STDDEV' else std_func.format(field)
                field_aggregates.append(aggregate)
        sql += ',\n'.join(field_aggregates)
        sql += f'\nFROM {_SQLiteDataProxy._DATA_TABLE}'
        results = self._read_sql(sql).squeeze()
        reshaped_results = pd.DataFrame()
        n_fields = len(fields)
        for i, sql_func, index in zip(range(0, len(results), n_fields), sql_funcs, _SUMMARY_STATS):
            next_row = results.iloc[i: i + n_fields]
            next_row.index = [col.replace(sql_func, '').replace('(', '').replace(')', '') for col in next_row.index]
            reshaped_results.loc[:, index] = next_row
        return reshaped_results

    def load_code_block_names(self) -> list[str]:
        sql = f'SELECT DISTINCT code_block_name FROM {_SQLiteDataProxy._DATA_TABLE}'
        return sorted(self._read_sql(sql).code_block_name)
