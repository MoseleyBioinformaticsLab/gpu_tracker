import typing as typ
import gpu_tracker as gput
import pytest as pt


@pt.fixture(name='use_context_manager', params=[True, False])
def get_use_context_manager(request) -> bool:
    yield request.param


test_tracker_data = [
    (
        True, 1697450.0, 5800.0, 300.0, 'bytes', 'megabytes', 'seconds',
        'Max RAM: 1697450.000 bytes\nMax GPU: 5800.000 megabytes\nCompute time: 300.000 seconds'
    ),
    (
        True, 1697.450, 5.8, 5.0, 'kilobytes', 'gigabytes', 'minutes',
        'Max RAM: 1697.450 kilobytes\nMax GPU: 5.800 gigabytes\nCompute time: 5.000 minutes'
    ),
    (
        False, 0.5505, 1600000.0, 300.0 / 3600, 'megabytes', 'kilobytes', 'hours',
        'Max RAM: 0.550 megabytes\nMax GPU: 1600000.000 kilobytes\nCompute time: 0.083 hours'
    ),
    (
        False, 550.5, 1600000000.0, 300.0 / (3600 * 24), 'kilobytes', 'bytes', 'days',
        'Max RAM: 550.500 kilobytes\nMax GPU: 1600000000.000 bytes\nCompute time: 0.003 days'
    )
]


@pt.mark.parametrize(
    'include_children,expected_ram,expected_gpu_ram,expected_time,ram_unit,gpu_ram_unit,time_unit,tracker_str', test_tracker_data)
def test_tracker(
        mocker, use_context_manager: bool, include_children: bool, expected_ram: float, expected_gpu_ram: float, expected_time: float,
        ram_unit: str, gpu_ram_unit: str, time_unit: str, tracker_str):
    class EventMock:
        def __init__(self):
            self.count = 0
            self.is_set = mocker.MagicMock(wraps=self.is_set)
            self.set = mocker.MagicMock()

        def is_set(self) -> bool:
            self.count += 1
            return self.count > 3

    class ThreadMock:
        def __init__(self, target: typ.Callable):
            self.target = target
            self.start = mocker.MagicMock(wraps=self.start)
            self.join = mocker.MagicMock()
            self.is_alive = mocker.MagicMock(return_value=False)

        def start(self):
            self.target()

    EventMock = mocker.patch('gpu_tracker.tracker.thrd.Event', wraps=EventMock)
    ThreadMock = mocker.patch('gpu_tracker.tracker.thrd.Thread', wraps=ThreadMock)
    process_id = 12
    child1_id = 21
    child2_id = 22
    process_rams = [440400, 440400, 550500]
    child1_rams = [590900, 590990, 490900]
    child2_rams = [666000, 666060, 333000]
    getpid_mock = mocker.patch('gpu_tracker.tracker.os.getpid', return_value=process_id)

    def get_process_mock(pid: int, rams: list[int], children: list[mocker.MagicMock] | None = None) -> mocker.MagicMock:
        return mocker.MagicMock(
            pid=pid,
            memory_info=mocker.MagicMock(side_effect=[mocker.MagicMock(rss=ram) for ram in rams]),
            children=mocker.MagicMock(return_value=children) if children is not None else None)

    child1_mock = get_process_mock(pid=child1_id, rams=child1_rams)
    child2_mock = get_process_mock(pid=child2_id, rams=child2_rams)
    process_mock = get_process_mock(pid=process_id, rams=process_rams, children=[child1_mock, child2_mock])
    ProcessMock = mocker.patch('gpu_tracker.tracker.psutil.Process', return_value=process_mock)
    nvidia_smi_outputs = [
        b'',
        b'12,1600 MiB\n21,700 MiB\n22,200 MiB',
        b'12,1500 MiB\n21,2100 MiB\n22,2200 MiB']
    check_output_mock = mocker.patch('gpu_tracker.tracker.subp.check_output', side_effect=nvidia_smi_outputs)
    time_mock = mocker.patch('gpu_tracker.tracker.time.time', side_effect=[800, 900, 1000, 1100])
    sleep_mock = mocker.patch('gpu_tracker.tracker._testable_sleep')
    sleep_time = 1.5
    join_timeout = 5.5
    if use_context_manager:
        with gput.Tracker(
                include_children=include_children, sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit,
                gpu_ram_unit=gpu_ram_unit, time_unit=time_unit) as tracker:
            pass
    else:
        tracker = gput.Tracker(
            include_children=include_children, sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit,
            gpu_ram_unit=gpu_ram_unit, time_unit=time_unit)
        tracker.start()
        tracker.stop()
    EventMock.assert_called_once_with()
    ThreadMock.assert_called_once_with(target=tracker._profile)
    tracker._thread.start.assert_called_once_with()
    _assert_args_list(mock=tracker._stop_event.is_set, expected_args_list=[()] * 4)
    _assert_args_list(mock=getpid_mock, expected_args_list=[()])
    _assert_args_list(mock=ProcessMock, expected_args_list=[(process_id,)] * 3)
    _assert_args_list(mock=process_mock.memory_info, expected_args_list=[()] * 3)
    _assert_args_list(mock=process_mock.children, expected_args_list=[()] * 3 if include_children else [])
    _assert_args_list(mock=child1_mock.memory_info, expected_args_list=[()] * 3 if include_children else [])
    _assert_args_list(mock=child2_mock.memory_info, expected_args_list=[()] * 3 if include_children else [])
    assert len(check_output_mock.call_args_list) == 3
    _assert_args_list(mock=time_mock, expected_args_list=[()] * 4)
    _assert_args_list(mock=sleep_mock, expected_args_list=[(sleep_time,)] * 3)
    assert tracker.max_ram == expected_ram
    assert tracker.max_gpu_ram == expected_gpu_ram
    assert tracker.compute_time == expected_time
    assert str(tracker) == tracker_str
    tracker._stop_event.set.assert_called_once_with()
    tracker._thread.join.assert_called_once_with(timeout=join_timeout)
    _assert_args_list(mock=tracker._thread.is_alive, expected_args_list=[()] * 2)


def _assert_args_list(mock, expected_args_list: list[tuple | dict], use_kwargs: bool = False):
    actual_args_list = [call.kwargs if use_kwargs else call.args for call in mock.call_args_list]
    assert actual_args_list == expected_args_list


@pt.mark.parametrize('kill_if_join_fails', [True, False])
def test_warnings(mocker, kill_if_join_fails: bool, caplog):
    n_join_attempts = 3
    join_timeout = 5.2
    mocker.patch('gpu_tracker.tracker.thrd.Event', return_value=mocker.MagicMock(set=mocker.MagicMock()))
    mocker.patch(
        'gpu_tracker.tracker.thrd.Thread',
        return_value=mocker.MagicMock(start=mocker.MagicMock(), is_alive=mocker.MagicMock(return_value=True), join=mocker.MagicMock())
    )
    exit_mock = mocker.patch('gpu_tracker.tracker.sys.exit')
    with gput.Tracker(kill_if_join_fails=kill_if_join_fails, n_join_attempts=n_join_attempts, join_timeout=join_timeout) as tracker:
        pass
    _assert_args_list(mock=tracker._stop_event.set, expected_args_list=[()] * n_join_attempts)
    _assert_args_list(mock=tracker._thread.join, expected_args_list=[{'timeout': join_timeout}] * n_join_attempts, use_kwargs=True)
    _assert_args_list(mock=tracker._thread.is_alive, expected_args_list=[()] * (n_join_attempts + 1))
    expected_warnings = ['Thread is still alive after join timout. Attempting to join again...'] * n_join_attempts
    expected_warnings.append(
        'Thread is still alive after 3 attempts to join. The thread will likely not end until the parent process ends.')
    if kill_if_join_fails:
        expected_warnings.append('The thread failed to join and kill_if_join_fails is set. Exiting ...')
        exit_mock.assert_called_once_with(1)
    else:
        assert not exit_mock.called
    for expected_warning, record in zip(expected_warnings, caplog.records):
        assert record.levelname == 'WARNING'
        assert record.message == expected_warning


def test_validate_unit():
    with pt.raises(ValueError) as error:
        gput.Tracker(ram_unit='milibytes')
    assert str(error.value) == '"milibytes" is not a valid memory unit. Valid values are bytes, gigabytes, kilobytes, megabytes, terabytes'
