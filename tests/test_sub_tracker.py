import pytest as pt
import gpu_tracker as gput
import utils


@pt.fixture(name='code_block_name', params=['my-code-block', None])
def get_code_block_name(request) -> str | None:
    yield request.param


@pt.fixture(name='sub_tracking_file', params=['sub-tracking-file.csv', 'sub-tracking-file.sqlite', None])
def get_sub_tracking_file(request) -> str | None:
    yield request.param


def test_sub_tracker(mocker, code_block_name: str | None, sub_tracking_file: str | None):
    n_iterations = 5
    getpid_mock = mocker.patch('gpu_tracker.sub_tracker.os.getpid', side_effect=[1234] * n_iterations)
    time_mock = mocker.patch(
        'gpu_tracker.sub_tracker.time', time=mocker.MagicMock(side_effect=range(n_iterations * 2)))
    default_code_block_end = 'test_sub_tracker.py:23'
    for _ in range(n_iterations):
        with gput.SubTracker(code_block_name=code_block_name, sub_tracking_file=sub_tracking_file) as sub_tracker:
            if code_block_name is None:
                assert sub_tracker.code_block_name.endswith(default_code_block_end)
            if sub_tracking_file is None:
                assert sub_tracker.sub_tracking_file == '1234.csv'
    if sub_tracking_file is None:
        assert len(getpid_mock.call_args_list) == n_iterations
    assert len(time_mock.time.call_args_list) == n_iterations * 2

    def code_block_name_test(val: str):
        if code_block_name is None:
            assert val.endswith(default_code_block_end)
        else:
            assert val == code_block_name
    utils.test_tracking_file(
        actual_tracking_file=sub_tracker.sub_tracking_file, expected_tracking_file='tests/data/sub-tracker.csv',
        excluded_col='code_block_name', excluded_col_test=code_block_name_test
    )


@pt.fixture(name='code_block_attribute', params=['my-attribute', None])
def get_code_block_attribute(request):
    yield request.param


def test_decorator(mocker, code_block_name: str | None, code_block_attribute: str | None):
    @gput.sub_track(code_block_name=code_block_name, code_block_attribute=code_block_attribute)
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
        actual_tracking_file='1234.csv', expected_tracking_file=f'tests/data/decorated-function.csv',
        excluded_col='code_block_name', excluded_col_test=code_block_name_test
    )
    if code_block_name is None and code_block_attribute is None:
        return_val = utils.function_in_other_file(1, 2, 3, kw1=4, kw2=5)
        assert return_val == ((1, 2, 3), {'kw1': 4, 'kw2': 5})
        assert len(getpid_mock.call_args_list) == n_iterations * 2 + 1
        assert len(time_mock.time.call_args_list) == n_iterations * 2 * 2 + 2

        def code_block_name_test(val):
            assert val.endswith('utils.py:function_in_other_file')
        utils.test_tracking_file(
            actual_tracking_file='1234.csv', expected_tracking_file='tests/data/decorated-function-other-file.csv',
            excluded_col='code_block_name', excluded_col_test=code_block_name_test
        )
