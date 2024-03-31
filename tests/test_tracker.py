import typing as typ
import gpu_tracker as gput
import json
import pytest as pt


@pt.fixture(name='operating_system', params=['Linux', 'not-linux'])
def get_operating_system(request) -> str:
    yield request.param


@pt.fixture(name='use_context_manager', params=[True, False])
def get_use_context_manager(request) -> bool:
    yield request.param


test_tracker_data = [
    ('bytes', 'megabytes', 'seconds'),
    ('kilobytes', 'gigabytes', 'minutes'),
    ('megabytes', 'kilobytes', 'hours'),
    ('kilobytes', 'bytes', 'days')
]


@pt.mark.parametrize('ram_unit,gpu_ram_unit,time_unit', test_tracker_data)
def test_tracker(mocker, use_context_manager: bool, operating_system: str, ram_unit: str, gpu_ram_unit: str, time_unit: str):
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

    system_mock = mocker.patch('gpu_tracker.tracker.platform.system', return_value=operating_system) 
    EventMock = mocker.patch('gpu_tracker.tracker.thrd.Event', wraps=EventMock)
    ThreadMock = mocker.patch('gpu_tracker.tracker.thrd.Thread', wraps=ThreadMock)
    process_id = 12
    child1_id = 21
    child2_id = 22
    process_rams = [440400, 440400, 550500]
    process_private_dirty = [[1847, 900], [1024, 789, 1941], [1337, 169, 1886, 934]]
    process_private_clean = [[1265, 180], [1031, 416, 460], [1078, 614, 101, 599]] 
    process_shared_dirty = [[963, 1715], [1424, 1101, 486], [545, 495, 911, 1690]]
    process_shared_clean = [[934, 1327], [337, 1523, 1473], [1553, 1963, 1008, 1871]]
    process_paths = [['heap', 'stack'], ['heap', 'stack', 'so1'], ['heap', 'stack', 'so1', 'so2']]
    child1_rams = [590900, 590990, 490900]
    child1_private_dirty = [[1703, 1174], [977, 383, 225], [1715, 1083, 453, 582]]
    child1_private_clean = [[449, 1652], [505, 376, 90], [431, 109, 1887, 423]]
    child1_shared_dirty = [[1767, 1528], [759, 1148, 1111], [953, 897, 1932, 1287]]
    child1_shared_clean = [[21, 1110], [650, 64, 1743], [1880, 1414, 1573, 600]]
    child1_paths = [['heap', 'stack'], ['heap', 'stack', 'so2'], ['heap', 'stack', 'so3', 'so4']]
    child2_rams = [666000, 666060, 333000]
    child2_private_dirty = [[782, 176], [851, 48, 1337], [1195, 1987, 651, 153]]
    child2_private_clean = [[931, 983], [1783, 1309, 965], [20, 714, 605, 1810]]
    child2_shared_dirty = [[79, 408], [537, 1517, 1937], [1908, 405, 1126, 1436]]
    child2_shared_clean = [[376, 1412], [1621, 241, 734], [1565, 1348, 1877, 775]]
    child2_paths = [['heap', 'stack'], ['heap', 'stack', 'so3'], ['heap', 'stack', 'so5', 'so6']]
    getpid_mock = mocker.patch('gpu_tracker.tracker.os.getpid', return_value=process_id)

    def get_process_mock(
            pid: int, rams: list[int], private_dirty: list[list[int]], private_clean: list[list[int]], shared_dirty: list[list[int]],
            shared_clean: list[list[int]], paths: list[list[str]], children: list[mocker.MagicMock] | None = None) -> mocker.MagicMock:
        memory_maps_side_effect = list[list[mocker.MagicMock()]]()
        for private_dirty, private_clean, shared_dirty, shared_clean, paths in zip(
                private_dirty, private_clean, shared_dirty, shared_clean, paths):
            memory_map_mocks = list[mocker.MagicMock]()
            for private_dirty, private_clean, shared_dirty, shared_clean, path in zip(
                    private_dirty, private_clean, shared_dirty, shared_clean, paths):
                memory_map_mock = mocker.MagicMock(
                    private_dirty=private_dirty, private_clean=private_clean, shared_dirty=shared_dirty, shared_clean=shared_clean,
                    path=path)
                memory_map_mocks.append(memory_map_mock)
            memory_maps_side_effect.extend([memory_map_mocks, memory_map_mocks])
        rams = [ram for ram in rams for _ in range(2)]
        return mocker.MagicMock(
            pid=pid,
            memory_info=mocker.MagicMock(side_effect=[mocker.MagicMock(rss=ram) for ram in rams]),
            memory_maps=mocker.MagicMock(side_effect=memory_maps_side_effect),
            children=mocker.MagicMock(return_value=children) if children is not None else None)

    child1_mock = get_process_mock(
        pid=child1_id, rams=child1_rams, private_dirty=child1_private_dirty, private_clean=child1_private_clean,
        shared_dirty=child1_shared_dirty, shared_clean=child1_shared_clean, paths=child1_paths)
    child2_mock = get_process_mock(
        pid=child2_id, rams=child2_rams, private_dirty=child2_private_dirty, private_clean=child2_private_clean,
        shared_dirty=child2_shared_dirty, shared_clean=child2_shared_clean, paths=child2_paths)
    process_mock = get_process_mock(
        pid=process_id, rams=process_rams, private_dirty=process_private_dirty, private_clean=process_private_clean,
        shared_dirty=process_shared_dirty, shared_clean=process_shared_clean, paths=process_paths, children=[child1_mock, child2_mock])
    ProcessMock = mocker.patch('gpu_tracker.tracker.psutil.Process', return_value=process_mock)
    virtual_memory_mock = mocker.patch(
        'gpu_tracker.tracker.psutil.virtual_memory', side_effect=[
            mocker.MagicMock(total=67 * 1e9), mocker.MagicMock(used=30 * 1e9), mocker.MagicMock(used=31 * 1e9),
            mocker.MagicMock(used=29 * 1e9)])
    nvidia_smi_outputs = [
        b'',
        b'12,1600 MiB\n21,700 MiB\n22,200 MiB',
        b'12,1500 MiB\n21,2100 MiB\n22,2200 MiB']
    check_output_mock = mocker.patch('gpu_tracker.tracker.subp.check_output', side_effect=nvidia_smi_outputs)
    time_mock = mocker.patch('gpu_tracker.tracker.time.time', side_effect=[800, 900, 1000, 1100])
    sleep_mock = mocker.patch('gpu_tracker.tracker._testable_sleep')
    log_spy = mocker.spy(gput.tracker.log, 'warning')
    sleep_time = 1.5
    join_timeout = 5.5
    if use_context_manager:
        with gput.Tracker(
                sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit, gpu_ram_unit=gpu_ram_unit,
                time_unit=time_unit) as tracker:
            pass
    else:
        tracker = gput.Tracker(
            sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit, gpu_ram_unit=gpu_ram_unit, time_unit=time_unit)
        tracker.start()
        tracker.stop()
    assert not log_spy.called
    _assert_args_list(virtual_memory_mock, [()] * 4)
    system_mock.assert_called_once_with()
    EventMock.assert_called_once_with()
    ThreadMock.assert_called_once_with(target=tracker._profile)
    tracker._thread.start.assert_called_once_with()
    _assert_args_list(mock=tracker._stop_event.is_set, expected_args_list=[()] * 4)
    _assert_args_list(mock=getpid_mock, expected_args_list=[()])
    _assert_args_list(mock=ProcessMock, expected_args_list=[(process_id,)])
    _assert_args_list(mock=process_mock.children, expected_args_list=[{'recursive': True}] * 8, use_kwargs=True)
    if operating_system == 'Linux':
        _assert_args_list(mock=process_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
        _assert_args_list(mock=child1_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
        _assert_args_list(mock=child2_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
    else:
        _assert_args_list(mock=process_mock.memory_info, expected_args_list=[()] * 6)
        _assert_args_list(mock=child1_mock.memory_info, expected_args_list=[()] * 6)
        _assert_args_list(mock=child2_mock.memory_info, expected_args_list=[()] * 6)
    assert len(check_output_mock.call_args_list) == 3
    _assert_args_list(mock=time_mock, expected_args_list=[()] * 4)
    _assert_args_list(mock=sleep_mock, expected_args_list=[(sleep_time,)] * 3)
    tracker._stop_event.set.assert_called_once_with()
    tracker._thread.join.assert_called_once_with(timeout=join_timeout)
    _assert_args_list(mock=tracker._thread.is_alive, expected_args_list=[()] * 2)
    expected_measurements_file = f'tests/data/{use_context_manager}-{operating_system}-{ram_unit}-{gpu_ram_unit}-{time_unit}'
    with open(f'{expected_measurements_file}.txt', 'r') as file:
        expected_tracker_str = file.read()
        assert expected_tracker_str == str(tracker)
    with open(f'{expected_measurements_file}.json', 'r') as file:
        expected_measurements = json.load(file)
        assert expected_measurements == tracker.to_json()


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
