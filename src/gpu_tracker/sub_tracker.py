"""The ``sub_tracker`` module contains the ``SubTracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
from __future__ import annotations
import inspect
import os
import time
import functools
import pandas as pd
import dataclasses as dclass
import pickle as pkl
import logging as log
import typing as typ
from ._helper_classes import _DataProxy, _SubTrackerLog, _SUMMARY_STATS, _summary_stats


class SubTracker:
    """
    Context manager that logs to a file for the purposes of sub tracking a code block using the timestamps at which the codeblock begins and ends.
    Entering the context manager marks the beginning of the code block and exiting the context manager marks the end of the code block.
    At the beginning of the codeblock, the ``SubTracker`` logs a row to a tabular file (".csv" or ".sqlite") that includes the timestamp along with a name for the code block and an indication of whether it is the start or end of the code bock.
    This resulting file can be used alongside a tracking file created by a ``Tracker`` object for more granular analysis of specific code blocks.

    :ivar str code_block_name: The name of the code block being sub-tracked.
    :ivar str sub_tracking_file: The path to the file where the sub-tracking info is logged.
    """

    def __init__(
            self, code_block_name: str | None = None, code_block_attribute: str | None = None, sub_tracking_file: str | None = None,
            overwrite: bool = False):
        """
        :param code_block_name: The name of the code block within a ``Tracker`` context that is being sub-tracked. Defaults to the file path where the ``SubTracker`` context is started followed by a colon followed by the ``code_block_attribute``.
        :param code_block_attribute: Only used if ``code_block_name`` is ``None``. Defaults to the line number where the ``SubTracker`` context is started.
        :param sub_tracking_file: The path to the file to log the time stamps of the code block being sub-tracked. To avoid file lock errors when a sub-tracking file is created in multiple different processes (i.e. multiple processes attempting to access the same file at the same time), the sub-tracking file of each process must have a unique name. For example, the ID of the process where the SubTracker context is created. Defaults to this process ID as the file name and in CSV format. These files can be combined into one using the ``Analyzer.combine_sub_tracking_files`` function.
        :param overwrite: Whether to overwrite the ``sub_tracking_file`` if it already existed before the beginning of this tracking session.
        """
        if code_block_name is not None:
            self.code_block_name = code_block_name
        else:
            stack = inspect.stack()
            caller_frame = stack[1]
            file_path = os.path.relpath(caller_frame.filename)
            code_block_attribute = caller_frame.lineno if code_block_attribute is None else code_block_attribute
            self.code_block_name = f'{file_path}:{code_block_attribute}'
        self.process_id = os.getpid()
        if sub_tracking_file is None:
            sub_tracking_file = f'{self.process_id}.csv'
        self.sub_tracking_file = sub_tracking_file
        self._data_proxy = _DataProxy.create(self.sub_tracking_file, overwrite)

    def _log(self, code_block_position: _SubTrackerLog.CodeBlockPosition):
        sub_tracker_log = _SubTrackerLog(
            process_id=self.process_id, code_block_name=self.code_block_name, position=code_block_position.value, timestamp=time.time())
        self._data_proxy.write_data(sub_tracker_log)

    def __enter__(self):
        self._log(_SubTrackerLog.CodeBlockPosition.START)
        return self

    def __exit__(self, *_):
        self._log(_SubTrackerLog.CodeBlockPosition.STOP)


def sub_track(
        code_block_name: str | None = None, code_block_attribute: str | None = None, sub_tracking_file: str | None = None,
        overwrite: bool = False):
    """
    Decorator for sub tracking calls to a specified function. Creates a ``SubTracker`` context that wraps the function call.

    :param code_block_name: The ``code_block_name`` argument passed to the ``SubTracker``. Defaults to the file path where the decorated function is defined followed by a colon followed by the ``code_block_attribute``.
    :param code_block_attribute: The ``code_block_attribute`` argument passed to the ``SubTracker``. Defaults to the name of the decorated function.
    :param sub_tracking_file: the ``sub_tracking_file`` argument passed to the ``SubTracker``. Same default as the ``SubTracker`` constructor. If using the decorated function in multiprocessing, if you'd like to name it based on the ID of a child process for uniqueness, you may need to set the start method to "spawn" like so ``multiprocessing.set_start_method('spawn')``.
    :param overwrite: The ``overwrite`` argument passed to the ``SubTracker``.
    """

    def decorator(func):
        nonlocal code_block_name, code_block_attribute, sub_tracking_file, overwrite
        if code_block_name is None:
            stack = inspect.stack()
            caller_frame = stack[1]
            file_path = os.path.abspath(caller_frame.filename)
            code_block_attribute = func.__name__ if code_block_attribute is None else code_block_attribute
            code_block_name = f'{file_path}:{code_block_attribute}'

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal sub_tracking_file, overwrite
            with SubTracker(
                    code_block_name=code_block_name, code_block_attribute=code_block_attribute, sub_tracking_file=sub_tracking_file,
                    overwrite=overwrite):
                return_value = func(*args, **kwargs)
            return return_value
        return wrapper
    return decorator


class SubTrackingAnalyzer:
    """
    Analyzes the per-code block tracking data using a tracking file and sub tracking file in order to produce summary statistics of resource usage for each individual code block.
    """

    def __init__(self, tracking_file: str | None, sub_tracking_file: str):
        """
        :param tracking_file: Path to the file containing the resource usage at each timepoint collected by a ``Tracker`` object.
        :param sub_tracking_file: Path to the file containing the start/stop timestamps of each call to a code block collected by a ``SubTracker`` object.
        """
        self._tracking_proxy = _DataProxy.create(tracking_file)
        self._sub_tracking_proxy = _DataProxy.create(sub_tracking_file)

    def read_static_data(self) -> pd.Series:
        """
        Reads the static data from the tracking file, including the resource units of measurement and system capacities.

        :return: The static data.
        """
        return self._tracking_proxy.read_static_data()

    def load_code_block_names(self) -> list[str]:
        """
        Loads the list of the names of the code blocks that were sub-tracked.

        :return: The code block names.
        """
        return self._sub_tracking_proxy.load_code_block_names()

    def combine_sub_tracking_files(self, files: list[str]):
        """
        Combines multiple sub-tracking files, perhaps that came from multiple processes running simultaneously, into a single sub-tracking file.

        :param files: The list of sub-tracking files to combine. All must end in the same file extension i.e. either ".csv" or ".sqlite".
        """
        self._sub_tracking_proxy.combine_files(files)

    def load_timestamp_pairs(self, code_block_name: str) -> list[tuple[float, float]]:
        """
        Loads the pairs of start and stop timestamps for each call to a code block that was sub-tracked.

        :param code_block_name: The name of the code block to get timestamp pairs for.
        :return: List of timestamp pairs.
        """
        return self._sub_tracking_proxy.load_timestamp_pairs(code_block_name)

    def load_timepoints(self, timestamp_pairs: list[tuple[float, float]]) -> pd.DataFrame:
        """
        Loads the resource usage measurements at each timepoint tracked within the timestamp pairs of a given code block.

        :param timestamp_pairs: The list of start and stop timestamp pairs of the code block.
        :return: The timepoint measurements.
        """
        return self._tracking_proxy.load_timepoints(timestamp_pairs)

    def overall_timepoint_results(self) -> pd.DataFrame:
        """
        Computes summary statistics for resource measurements across all tracked timepoints as compared to an individual sub-tracked code block.

        :return: Summary statistics across all timepoints.
        """
        return self._tracking_proxy.overall_timepoint_results()

    def sub_tracking_results(self) -> SubTrackingResults:
        """
        Generates a detailed report including summary statistics for the overall resource usage across all timepoints as well as that of each code block that was sub-tracked.

        :return: A data object containing the overall summary statistics, summary statistics for each code block, the static data, etc.
        """
        overall_results = self.overall_timepoint_results()
        code_block_names = self.load_code_block_names()
        static_data = self.read_static_data()
        code_block_results = list[CodeBlockResults]()
        for code_block_name in code_block_names:
            time_stamp_pairs = self.load_timestamp_pairs(code_block_name)
            time_stamp_diffs = pd.Series([stop_time - start_time for (start_time, stop_time) in time_stamp_pairs])
            compute_time_results = _summary_stats(time_stamp_diffs)
            compute_time_results['total'] = time_stamp_diffs.sum().item()
            timepoints = self.load_timepoints(time_stamp_pairs)
            num_non_empty_calls = sum(
                [
                    any(
                        (timepoints.timestamp >= start_time) & (timepoints.timestamp <= stop_time)
                    ) for start_time, stop_time in time_stamp_pairs
                ]
            )
            timepoints = timepoints.drop(columns='timestamp')
            resource_usage = _summary_stats(timepoints).T
            code_block_results.append(
                CodeBlockResults(
                    name=code_block_name, num_timepoints=len(timepoints), num_calls=len(time_stamp_pairs),
                    num_non_empty_calls=num_non_empty_calls, compute_time=compute_time_results, resource_usage=resource_usage
                )
            )
        return SubTrackingResults(overall_results, static_data, code_block_results)


class TrackingComparison:
    """
    Compares multiple tracking sessions to determine differences in computational resource usage by loading sub-tracking results given their file paths.
    Sub-tracking results files must be in pickle format e.g. calling the ``SubTrackingAnalyzer.compare`` method and storing the returned ``SubTrackingResults`` in a pickle file.
    If code block results are not included in the sub-tracking files (i.e. no code blocks were sub-tracked), then only overall results are compared.
    Code blocks are compared by their name. If their name only differentiates by line number (i.e. their name is of the form <file-path:line-number>), then it's assumed that the same order of the code blocks is used even if the line numbers are different.
    This is useful to determine how resource usage changes based on differences in implementation, input data, etc.

    :ivar dict[str, SubTrackingResults] results_map: Mapping of the name of each tracking session to the ``SubTrackingResults`` of the corresponding tracking sessions. Can be used for a user-defined custom comparison.
    """
    def __init__(self, file_path_map: dict[str, str]):
        """
        :param file_path_map: Mapping of the name of each tracking session to the path of the pickle file containing the ``SubTrackingResults`` of the corresponding tracking sessions. Used to construct the ``results_map`` attribute.
        :raises ValueError: Raised if the code block results of each tracking session don't match.
        """
        for name in file_path_map.keys():
            self._name1 = name
            break
        self.results_map = dict[str, SubTrackingResults]()
        for name, file in file_path_map.items():
            with open(file, 'rb') as file:
                self.results_map[name] = pkl.load(file)
        results1 = self.results_map[self._name1]
        code_block_results1 = TrackingComparison._sort_code_block_results(results1)
        results1.code_block_results = code_block_results1
        for name2, result in self.results_map.items():
            if name2 == self._name1:
                continue
            results = self.results_map[name2]
            code_block_results2 = TrackingComparison._sort_code_block_results(results)
            if len(code_block_results1) != len(code_block_results2):
                raise ValueError(
                    f'All sub-tracking results must have the same number of code blocks. First has {len(code_block_results1)}'
                    f' code blocks but tracking session "{name2}" has {len(code_block_results2)} code blocks.'
                )
            for code_block_results1_, code_block_results2_ in zip(code_block_results1, code_block_results2):
                code_block_name1, code_block_name2 = code_block_results1_.name, code_block_results2_.name
                if code_block_name1 != code_block_name2:
                    line_num1 = TrackingComparison._get_line_num(code_block_name1)
                    line_num2 = TrackingComparison._get_line_num(code_block_name2)
                    if line_num1 is not None and line_num2 is not None:
                        if code_block_name1.split(':')[:-1] == code_block_name2.split(':')[:-1]:
                            log.warning(
                                f'Code block name "{code_block_name1}" of tracking session "{self._name1}" matched with code'
                                f' block name "{code_block_name2}" of tracking session "{name2}" but they differ by '
                                f'line number. If these code blocks were not meant to match, their comparison will not'
                                f' be valid and their names must be disambiguated.'
                            )
                            match = True
                        else:
                            match = False
                    else:
                        match = False
                else:
                    match = True
                if not match:
                    raise ValueError(
                        f'Code block name "{code_block_name1}" of tracking session "{self._name1}" does not match code'
                        f' block name "{code_block_name2}" of tracking session "{name2}"'
                    )
            results.code_block_results = code_block_results2

    def compare(self, statistic: str = 'mean') -> ComparisonResults:
        """
        Performs the comparison between tracking sessions, comparing both the code block results and the overall results.
        :param statistic: The summary statistic of the measurements to compare. One of 'min', 'max', 'mean', or 'std'.
        :return: The results of the comparison including the overall resource usage, the resource usage of the code blocks, and the compute time of the code blocks for each tracking session.
        """
        if statistic not in _SUMMARY_STATS:
            raise ValueError(
                f"Invalid summary statistic '{statistic}'. Valid values are {' '.join(_SUMMARY_STATS)}."
            )
        results1 = self.results_map[self._name1]
        overall_resource_usages = dict[str, pd.Series]()
        code_block_resource_usages = dict[str, dict[str, pd.Series]]()
        for measurement in results1.overall.index:
            overall_comparison = {name: results.overall[statistic][measurement].item() for name, results in self.results_map.items()}
            overall_resource_usages[measurement] = pd.Series(overall_comparison).sort_values(ascending=True)
            if results1.code_block_results:
                code_block_resource_usages[measurement] = TrackingComparison._get_code_block_comparisons(
                    self.results_map, lambda code_block_result: code_block_result.resource_usage[statistic][measurement].item()
                )
        code_block_compute_times = TrackingComparison._get_code_block_comparisons(
            self.results_map, lambda code_block_result: code_block_result.compute_time[statistic].item()
        ) if results1.code_block_results else dict()
        return ComparisonResults(
            overall_resource_usage=overall_resource_usages, code_block_resource_usage=code_block_resource_usages,
            code_block_compute_time=code_block_compute_times
        )

    @staticmethod
    def _sort_code_block_results(results: SubTrackingResults) -> list[CodeBlockResults]:
        max_line_num_len = 0
        for code_block_results in results.code_block_results:
            line_num = TrackingComparison._get_line_num(code_block_results.name)
            if line_num is not None:
                max_line_num_len = max(max_line_num_len, len(line_num))
        return sorted(
            results.code_block_results, key=lambda r: TrackingComparison._sort_code_block_name(r.name, max_line_num_len)
        )

    @staticmethod
    def _sort_code_block_name(name: str, max_line_num_len: int) -> str:
        line_num = TrackingComparison._get_line_num(name)
        if line_num is not None:
            line_num = line_num.zfill(max_line_num_len)
            name = ':'.join(name.split(':')[:-1] + [line_num])
        return name

    @staticmethod
    def _get_line_num(code_block_name: str) -> str | None:
        if ':' in code_block_name:
            line_num = code_block_name.split(':')[-1]
            try:
                int(line_num)
                return line_num
            except ValueError:
                return None
        return None

    @staticmethod
    def _get_code_block_comparisons(name_to_results: dict[str, SubTrackingResults], get_statistic: typ.Callable) -> dict[str, pd.Series]:
        code_block_comparisons = dict[str, pd.Series]()
        for matching_code_block_results in zip(
                *[
                    [
                        (name, code_block_results) for code_block_results in results.code_block_results
                    ] for name, results in name_to_results.items()
                ]
        ):
            code_block_name = f'{" -> ".join({code_block_results.name for _, code_block_results in matching_code_block_results})}'
            code_block_comparison = {
                name: get_statistic(code_block_results) for name, code_block_results in matching_code_block_results
            }
            code_block_comparison = pd.Series(code_block_comparison).sort_values(ascending=True)
            code_block_comparisons[code_block_name] = code_block_comparison
        return code_block_comparisons


@dclass.dataclass
class CodeBlockResults:
    """
    Results of a particular code block that was sub-tracked.

    :param name: The name of the code block.
    :param num_timepoints: The number of timepoints tracked across all calls to the code block.
    :param num_calls: The number times the code block was called / executed.
    :param num_non_empty_calls: The number code block calls with at least one timepoint tracked within the start / stop time.
    :param compute_time: Compute time measurements for the code block including the total time spent running this code block, the average time between the start / stop time, etc.
    :param resource_usage: Summary statistics for the resource usage during the times the code block was called i.e. in between all its start / stop times
    """
    name: str
    num_timepoints: int
    num_calls: int
    num_non_empty_calls: int
    compute_time: pd.Series
    resource_usage: pd.DataFrame


@dclass.dataclass
class SubTrackingResults:
    """
    Comprehensive results for a tracking session including resource usage measurements for individual code blocks.

    :param overall: The overall summary statistics across all timepoints tracked.
    :param static_data: The static data measured during a tracking session.
    :param code_block_results: Results for individual code blocks including summary statistics for the timepoints within each code block.
    """
    overall: pd.DataFrame
    static_data: pd.Series
    code_block_results: list[CodeBlockResults]

    def to_json(self) -> dict:
        """
        Converts the sub-tracking results into JSON format.

        :return: The JSON version of the sub-tracking results.
        """
        results = dclass.asdict(self)
        results['overall'] = _dataframe_to_json(results['overall'])
        results['static_data'] = results['static_data'].to_dict()
        for code_block_result in results['code_block_results']:
            code_block_result['compute_time'] = code_block_result['compute_time'].to_dict()
            code_block_result['resource_usage'] = _dataframe_to_json(code_block_result['resource_usage'])
        return results

    def __str__(self) -> str:
        """
        Converts the sub-tracking results to text format.

        :return: The string representation of the sub-tracking results.
        """
        results = dclass.asdict(self)
        return _dict_to_str('', results, 0)


def _dataframe_to_json(df: pd.DataFrame) -> dict:
    result = dict()
    for index, row in df.iterrows():
        result[index] = row.to_dict()
    return result


def _dict_to_str(string: str, results: dict, indent: int, no_title_keys: set[str] | None = None) -> str:
    indent_str = '\t' * indent
    results = {
        (
            f'{indent_str}{key.replace("_", " ").title() if no_title_keys is None or key not in no_title_keys else key}'
        ): value for key, value in results.items()
    }
    max_key_len = max(len(key) for key in results.keys())
    for key, value in results.items():
        if type(value) is pd.Series:
            value = value.to_frame().T
            value = value.rename({value.index[0]: ''})
        if type(value) is dict:
            string += f'{key}:\n'
            string = _dict_to_str(string, value, indent + 1, no_title_keys)
        elif type(value) is pd.DataFrame:
            string += f'{key}:\n'
            with pd.option_context(
                    'display.max_rows', None, 'display.max_columns', None, 'display.width', 5000, 'display.float_format',
                    lambda x: f'{float(f"{x:.8f}")}'
            ):
                df_str = str(value)
            df_str = '\n'.join(indent_str + '\t' + line for line in df_str.splitlines())
            string += df_str + '\n'
        elif type(value) is list:
            string += f'{key}:\n'
            for value in value:
                string = _dict_to_str(string, value, indent + 1, no_title_keys) + '\n'
        else:
            value = f'{value:.8f}' if type(value) is float else value
            string += f'{key}:{" " * (max_key_len - len(key))} {value}\n'
    return string


@dclass.dataclass
class ComparisonResults:
    """
    Contains the comparison of the measurements of multiple tracking sessions provided by the ``TrackingComparison`` class's ``compare`` method.

    :param overall_resource_usage: For each measurement, compares the resource usage across tracking sessions.
    :param code_block_resource_usage: For each measurement and for each code block, compares the resource usage of the code block across tracking sessions.
    :param code_block_compute_time: For each code block, compares the compute time of the code block across tracking sessions.
    """
    overall_resource_usage: dict[str, pd.Series]
    code_block_resource_usage: dict[str, dict[str, pd.Series]]
    code_block_compute_time: dict[str, pd.Series]

    def to_json(self) -> dict:
        """
        Converts the tracking comparison results into JSON format.

        :return: The JSON version of the comparison results.
        """
        results = dclass.asdict(self)
        results['overall_resource_usage'] = ComparisonResults._comparisons_to_dict(results['overall_resource_usage'])
        for measurement, comparisons in results['code_block_resource_usage'].items():
            results['code_block_resource_usage'][measurement] = ComparisonResults._comparisons_to_dict(comparisons)
        results['code_block_compute_time'] = ComparisonResults._comparisons_to_dict(results['code_block_compute_time'])
        return results

    @staticmethod
    def _comparisons_to_dict(comparisons: dict[str, pd.Series]) -> dict:
        return {name: comparison.to_dict() for name, comparison in comparisons.items()}

    def __str__(self) -> str:
        """
        Converts the tracking comparison results to text format.

        :return: The string representation of the comparison results.
        """
        results = dclass.asdict(self)
        return _dict_to_str('', results, 0, no_title_keys=set(name for name in self.code_block_compute_time.keys()))
