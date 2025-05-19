import pandas as pd
import sqlalchemy as sqlalc
import os
# noinspection PyProtectedMember
from gpu_tracker._helper_classes import _SQLiteDataProxy
import gpu_tracker as gput


def assert_args_list(mock, expected_args_list: list[tuple | dict], use_kwargs: bool = False):
    actual_args_list = [call.kwargs if use_kwargs else call.args for call in mock.call_args_list]
    assert actual_args_list == expected_args_list


def _assert_warnings(caplog, expected_warnings: list[str]):
    for expected_warning, record in zip(expected_warnings, caplog.records):
        assert record.levelname == 'WARNING'
        assert record.message == expected_warning


def test_tracking_file(
        actual_tracking_file: str, expected_tracking_file: str, excluded_col: str | None = None, excluded_col_test=None,
        is_sub_tracking: bool = False):
    if actual_tracking_file.endswith('.csv'):
        if is_sub_tracking:
            actual_tracking_log = pd.read_csv(actual_tracking_file)
        else:
            actual_tracking_log = pd.read_csv(actual_tracking_file, skiprows=2)
    else:
        engine = sqlalc.create_engine(f'sqlite:///{actual_tracking_file}', poolclass=sqlalc.pool.NullPool)
        actual_tracking_log = pd.read_sql_table(_SQLiteDataProxy._DATA_TABLE, engine)
    if excluded_col is not None:
        actual_tracking_log[excluded_col].apply(excluded_col_test)
        actual_tracking_log = actual_tracking_log[actual_tracking_log.columns.difference([excluded_col])]
    expected_tracking_log = pd.read_csv(expected_tracking_file)
    pd.testing.assert_frame_equal(expected_tracking_log, actual_tracking_log, atol=1e-10, rtol=1e-10)
    os.remove(actual_tracking_file)


@gput.sub_track()
def function_in_other_file(*args, **kwargs):
    return args, kwargs
