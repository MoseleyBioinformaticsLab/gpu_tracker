"""The ``sub_tracker`` module contains the ``SubTracker`` class which can alternatively be imported directly from the ``gpu_tracker`` package."""
import inspect
import os
import time
import functools
from ._helper_classes import _Writer, _SubTrackerLog


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
            self, code_block_name: str | None = None, code_block_attribute: str | None = None, sub_tracking_file: str | None = None):
        """
        :param code_block_name: The name of the code block within a ``Tracker`` context that is being sub-tracked. Defaults to the file path followed by a colon followed by the ``code_block_attribute``.
        :param code_block_attribute: Only used if ``code_block_name`` is ``None``. Defaults to the line number where the SubTracker context is started.
        :param sub_tracking_file: The path to the file to log the time stamps of the code block being sub-tracked. Defaults to the ID of the process where the SubTracker context is created and in CSV format.
        """
        if code_block_name is not None:
            self.code_block_name = code_block_name
        else:
            stack = inspect.stack()
            caller_frame = stack[1]
            file_path = os.path.abspath(caller_frame.filename)
            code_block_attribute = caller_frame.lineno if code_block_attribute is None else code_block_attribute
            self.code_block_name = f'{file_path}:{code_block_attribute}'
        if sub_tracking_file is None:
            sub_tracking_file = f'{os.getpid()}.csv'
        self.sub_tracking_file = sub_tracking_file
        self._sub_tracking_file = _Writer.create(self.sub_tracking_file)

    def _log(self, code_block_position: _SubTrackerLog.CodeBlockPosition):
        sub_tracker_log = _SubTrackerLog(
            code_block_name=self.code_block_name, position=code_block_position.value, timestamp=time.time())
        self._sub_tracking_file.write_row(sub_tracker_log)

    def __enter__(self):
        self._log(_SubTrackerLog.CodeBlockPosition.START)
        return self

    def __exit__(self, *_):
        self._log(_SubTrackerLog.CodeBlockPosition.STOP)


def sub_track(code_block_name: str | None = None, code_block_attribute: str | None = None, sub_tracking_file: str | None = None):
    """
    Decorator for sub tracking calls to a specified function.

    :param code_block_name: The name of the code block within a ``Tracker`` context that is being sub-tracked. Defaults to the file path followed by a colon followed by the ``code_block_attribute``.
    :param code_block_attribute: Only used if ``code_block_name`` is ``None``. Defaults to the name of the decorated function i.e. the function being sub-tracked.
    :param sub_tracking_file: The path to the file to log the time stamps of the code block being sub-tracked. Defaults to the ID of the process where the SubTracker context is created and in CSV format.
    """
    def decorator(func):
        nonlocal code_block_name, code_block_attribute, sub_tracking_file
        if code_block_name is None:
            stack = inspect.stack()
            caller_frame = stack[1]
            file_path = os.path.abspath(caller_frame.filename)
            code_block_attribute = func.__name__ if code_block_attribute is None else code_block_attribute
            code_block_name = f'{file_path}:{code_block_attribute}'

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal sub_tracking_file
            with SubTracker(
                    code_block_name=code_block_name, code_block_attribute=code_block_attribute, sub_tracking_file=sub_tracking_file
            ):
                return_value = func(*args, **kwargs)
            return return_value
        return wrapper
    return decorator
