import pytest as pt
import json
import os
import pandas as pd
import sqlalchemy as sqlalc
import deepdiff as deepd
import gpu_tracker as gput
import utils


@pt.fixture(name='code_block_name', params=['my-code-block', None])
def get_code_block_name(request) -> str | None:
    yield request.param


@pt.fixture(name='sub_tracking_file', params=['sub-tracking-file.csv', 'sub-tracking-file.sqlite'])
def get_sub_tracking_file(request) -> str | None:
    yield request.param


def test_sub_tracker(mocker, code_block_name: str | None, sub_tracking_file: str):
    sub_tracking_file = f'{code_block_name}_{sub_tracking_file}'
    n_iterations = 5
    getpid_mock = mocker.patch('gpu_tracker.sub_tracker.os.getpid', side_effect=[1234] * n_iterations)
    time_mock = mocker.patch(
        'gpu_tracker.sub_tracker.time', time=mocker.MagicMock(side_effect=range(n_iterations * 2)))
    default_code_block_end = 'test_sub_tracker.py:29'
    for _ in range(n_iterations):
        with gput.SubTracker(code_block_name=code_block_name, sub_tracking_file=sub_tracking_file) as sub_tracker:
            if code_block_name is None:
                assert sub_tracker.code_block_name.endswith(default_code_block_end)
    assert len(getpid_mock.call_args_list) == n_iterations
    assert len(time_mock.time.call_args_list) == n_iterations * 2

    def code_block_name_test(val: str):
        if code_block_name is None:
            assert val.endswith(default_code_block_end)
        else:
            assert val == code_block_name
    utils.test_tracking_file(
        actual_tracking_file=sub_tracker.sub_tracking_file, expected_tracking_file='tests/data/sub-tracker.csv',
        excluded_col='code_block_name', excluded_col_test=code_block_name_test, is_sub_tracking=True
    )


@pt.fixture(name='code_block_attribute', params=['my-attribute', None])
def get_code_block_attribute(request):
    yield request.param


def test_decorator(mocker, code_block_name: str | None, code_block_attribute: str | None):
    sub_tracking_file = f'{code_block_name}_{code_block_attribute}.csv'

    @gput.sub_track(code_block_name=code_block_name, code_block_attribute=code_block_attribute, sub_tracking_file=sub_tracking_file)
    def decorated_function(arg1: int, arg2: int, kwarg1: int = 1, kwarg2: int = 2) -> int:
        return arg1 + arg2 - (kwarg1 + kwarg2)
    getpid_mock = mocker.patch('gpu_tracker.sub_tracker.os.getpid', return_value=1234)
    n_iterations = 3
    time_mock = mocker.patch('gpu_tracker.sub_tracker.time', time=mocker.MagicMock(side_effect=range(n_iterations * 2 * 2 + 2)))
    for _ in range(n_iterations):
        return_val = decorated_function(2, 5)
        assert return_val == 4
        return_val = decorated_function(3, 2, kwarg1=2, kwarg2=1)
        assert return_val == 2
    assert len(getpid_mock.call_args_list) == n_iterations * 2
    assert len(time_mock.time.call_args_list) == n_iterations * 2 * 2

    def code_block_name_test(val):
        if code_block_name is None:
            if code_block_attribute is None:
                assert val.endswith('test_sub_tracker.py:decorated_function')
            else:
                assert val.endswith('test_sub_tracker.py:my-attribute')
        else:
            assert val == code_block_name
    utils.test_tracking_file(
        actual_tracking_file=sub_tracking_file, expected_tracking_file=f'tests/data/decorated-function.csv',
        excluded_col='code_block_name', excluded_col_test=code_block_name_test, is_sub_tracking=True
    )
    if code_block_name is None and code_block_attribute is None:
        return_val = utils.function_in_other_file(1, 2, 3, kw1=4, kw2=5)
        assert return_val == ((1, 2, 3), {'kw1': 4, 'kw2': 5})
        assert len(getpid_mock.call_args_list) == n_iterations * 2 + 1
        assert len(time_mock.time.call_args_list) == n_iterations * 2 * 2 + 2

        def code_block_name_test(val):
            assert val.endswith('utils.py:function_in_other_file')
        utils.test_tracking_file(
            actual_tracking_file=f'1234.csv', expected_tracking_file='tests/data/decorated-function-other-file.csv',
            excluded_col='code_block_name', excluded_col_test=code_block_name_test, is_sub_tracking=True
        )


@pt.fixture(name='format_', params=['csv', 'sqlite'])
def get_format(request):
    yield request.param


def test_analysis(format_: str):
    folder = 'tests/data/sub-tracking-results'
    tracking_file = f'{folder}/tracking.{format_}'
    sub_tracking_file = f'{folder}/sub-tracking.{format_}'
    analyzer = gput.SubTrackingAnalyzer(tracking_file, sub_tracking_file)
    actual_results = analyzer.sub_tracking_results()
    with open(f'{folder}/results.json', 'r') as file:
        expected_json_results = json.load(file)
    diff = deepd.DeepDiff(expected_json_results, actual_results.to_json(), significant_digits=12)
    assert not diff
    with open(f'{folder}/results.txt', 'r') as file:
        expected_str_results = file.read()
    assert expected_str_results == str(actual_results)


def test_combine(format_: str):
    folder = 'tests/data/sub-tracking-results'
    files = [f'{folder}/files-to-combine/{name}' for name in os.listdir(f'{folder}/files-to-combine') if name.endswith(format_)]
    with pt.raises(ValueError) as error:
        wrong_extension = "csv" if format_ == "sqlite" else "sqlite"
        invalid_file = f'wrong-extension.{wrong_extension}'
        analyzer = gput.SubTrackingAnalyzer(None, invalid_file)
        analyzer.combine_sub_tracking_files(files)
    assert str(error.value) == f'File {files[0]} does not end with the same extension as {invalid_file}. Must end in ".{wrong_extension}".'
    sub_tracking_file = f'combined.{format_}'
    analyzer = gput.SubTrackingAnalyzer(None, sub_tracking_file)
    analyzer.combine_sub_tracking_files(files)
    expected_path = f'{folder}/sub-tracking.{format_}'
    if format_ == 'csv':
        expected_results = pd.read_csv(expected_path)
        actual_results = pd.read_csv(sub_tracking_file)
    else:
        expected_results = pd.read_sql('data', sqlalc.create_engine(f'sqlite:///{expected_path}'))
        actual_results = pd.read_sql('data', sqlalc.create_engine(f'sqlite:///{sub_tracking_file}'))
    pd.testing.assert_frame_equal(expected_results, actual_results, atol=1e-10, rtol=1e-10)
    with pt.raises(ValueError) as error:
        analyzer.combine_sub_tracking_files(files)
    assert str(error.value) == f'Cannot create sub-tracking file {sub_tracking_file}. File already exists.'
    os.remove(sub_tracking_file)
