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
    expected_tracking_file = f'tests/data/{code_block_name}_{sub_tracking_file}.csv'
    utils.test_tracking_file(
        actual_tracking_file=sub_tracker.sub_tracking_file, expected_tracking_file=expected_tracking_file,
        excluded_col='code_block_name', excluded_col_test=code_block_name_test
    )
