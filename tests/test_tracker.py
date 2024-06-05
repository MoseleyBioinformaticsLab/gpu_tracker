import gpu_tracker as gput
import psutil
import json
import os
import pytest as pt
import utils

nvidia_smi_unavailable_message = 'The nvidia-smi command is not available. Please install the Nvidia drivers to track GPU usage. ' \
                                 'Otherwise the Max GPU RAM values will remain 0.0'


@pt.fixture(name='operating_system', params=['Linux', 'not-linux'])
def get_operating_system(request) -> str:
    yield request.param


@pt.fixture(name='use_context_manager', params=[True, False])
def get_use_context_manager(request) -> bool:
    yield request.param


def multiply_list(_list: list, multiple=2) -> list:
    return [item for item in _list for _ in range(multiple)]


test_tracker_data = [
    ('bytes', 'megabytes', 'seconds', None, 3),
    ('kilobytes', 'gigabytes', 'minutes', {'gpu-id1'}, 2),
    ('megabytes', 'kilobytes', 'hours', {'gpu-id1', 'gpu-id2'}, 1),
    ('kilobytes', 'bytes', 'days', {'gpu-id1', 'gpu-id2', 'gpu-id3'}, None)
]


@pt.mark.parametrize('ram_unit,gpu_ram_unit,time_unit,gpu_uuids,n_expected_cores', test_tracker_data)
def test_tracker(
        mocker, use_context_manager: bool, operating_system: str, ram_unit: str, gpu_ram_unit: str, time_unit: str, gpu_uuids: set[str],
        n_expected_cores: int):
    class EventMock:
        def __init__(self):
            self.count = 0
            self.is_set = mocker.MagicMock(wraps=self.is_set)
            self.set = mocker.MagicMock()

        def is_set(self) -> bool:
            self.count += 1
            return self.count > 3

    system_mock = mocker.patch('gpu_tracker.tracker.platform.system', return_value=operating_system)
    EventMock = mocker.patch('gpu_tracker.tracker.mproc.Event', wraps=EventMock)
    main_process_id = 12
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

    def get_process_mock(
            pid: int, rams: list[int], private_dirty: list[list[int]], private_clean: list[list[int]], shared_dirty: list[list[int]],
            shared_clean: list[list[int]], paths: list[list[str]], cpu_percentages: list[float], num_threads: list[float],
            children: list[mocker.MagicMock] | None = None) -> mocker.MagicMock:
        memory_maps_side_effect = list[list[mocker.MagicMock]]()
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
        rams = multiply_list(rams)
        cpu_percentages = multiply_list(cpu_percentages)
        num_threads = multiply_list(num_threads)
        return mocker.MagicMock(
            pid=pid,
            memory_info=mocker.MagicMock(side_effect=[mocker.MagicMock(rss=ram) for ram in rams]),
            memory_maps=mocker.MagicMock(side_effect=memory_maps_side_effect), cpu_percent=mocker.MagicMock(side_effect=cpu_percentages),
            num_threads=mocker.MagicMock(side_effect=num_threads),
            children=mocker.MagicMock(return_value=children) if children is not None else None)

    child1_mock = get_process_mock(
        pid=child1_id, rams=child1_rams, private_dirty=child1_private_dirty, private_clean=child1_private_clean,
        shared_dirty=child1_shared_dirty, shared_clean=child1_shared_clean, paths=child1_paths, cpu_percentages=[88.7, 90.2, 98.7],
        num_threads=[1, 2, 3])
    child2_mock = get_process_mock(
        pid=child2_id, rams=child2_rams, private_dirty=child2_private_dirty, private_clean=child2_private_clean,
        shared_dirty=child2_shared_dirty, shared_clean=child2_shared_clean, paths=child2_paths, cpu_percentages=[45.6, 22.5, 43.5],
        num_threads=[4, 2, 3])
    tracking_process_pid = 666
    tracking_process_mock = mocker.MagicMock(pid=tracking_process_pid)
    resource_tracker_pid = 13
    resource_tracker_mock = mocker.MagicMock(pid=resource_tracker_pid)
    main_process_mock = get_process_mock(
        pid=main_process_id, rams=process_rams, private_dirty=process_private_dirty, private_clean=process_private_clean,
        shared_dirty=process_shared_dirty, shared_clean=process_shared_clean, paths=process_paths, cpu_percentages=[60.4, 198.9, 99.8],
        num_threads=[0, 0, 2], children=[child1_mock, child2_mock, tracking_process_mock, resource_tracker_mock])
    child_mocks = [mocker.MagicMock(pid=pid) for pid in (child1_id, child2_id)]
    current_process_mock = mocker.MagicMock(
        children=mocker.MagicMock(side_effect=[child_mocks, [resource_tracker_mock] + child_mocks]))
    PsProcessMock = mocker.patch('gpu_tracker.tracker.psutil.Process', side_effect=[current_process_mock, main_process_mock])

    def start_mock(self):
        start_mock.called = True
        self.pid = tracking_process_pid
        self.run()

    start_mock.called = False
    mocker.patch.object(gput.tracker._TrackingProcess, 'start', new=start_mock)
    mocker.patch.object(gput.tracker._TrackingProcess, 'pid', new=None)
    mocker.patch.object(gput.tracker._TrackingProcess, 'join')
    mocker.patch.object(gput.tracker._TrackingProcess, 'is_alive', return_value=False)
    mocker.patch.object(gput.tracker._TrackingProcess, 'terminate')
    mocker.patch.object(gput.tracker._TrackingProcess, 'close')
    virtual_memory_mock = mocker.patch(
        'gpu_tracker.tracker.psutil.virtual_memory', side_effect=[
            mocker.MagicMock(total=67 * 1e9), mocker.MagicMock(used=30 * 1e9), mocker.MagicMock(used=31 * 1e9),
            mocker.MagicMock(used=29 * 1e9)])
    nvidia_smi_outputs = [
        b'',
        b' uuid,memory.total [MiB]\ngpu-id1,12198 MiB\ngpu-id2,12198 MiB\ngpu-id3 , 12198MiB',
        b'pid, used_gpu_memory [MiB]\n',
        b'uuid, memory.used [MiB], utilization.gpu [%]\ngpu-id1, 0 MiB, 0 %\ngpu-id2 , 0 MiB, 0 %\ngpu-id3 , 0 MiB, 0 %',
        b'pid, used_gpu_memory [MiB]\n12,1600 MiB\n21,700 MiB\n22,200 MiB',
        b'uuid, memory.used [MiB], utilization.gpu [%]\ngpu-id1, 1600 MiB,75 %\ngpu-id2,900 MiB , 50 %\n gpu-id3, 500 MiB, 25 %',
        b'pid, used_gpu_memory [MiB]\n12,1500 MiB\n21,2100 MiB\n22,2200 MiB',
        b'uuid, memory.used [MiB], utilization.gpu [%]\ngpu-id1, 1500 MiB, 55 %\n gpu-id2, 4300 MiB, 45%\ngpu-id3,700MiB,35%']
    check_output_mock = mocker.patch('gpu_tracker.tracker.subp.check_output', side_effect=nvidia_smi_outputs)
    cpu_count_mock = mocker.patch('gpu_tracker.tracker.psutil.cpu_count', return_value=4)
    cpu_percent_mock = mocker.patch(
        'gpu_tracker.tracker.psutil.cpu_percent', side_effect=[[67.5, 27.3, 77.8, 97.9], [57.6, 58.2, 23.5, 99.8], [78.3, 88.3, 87.2, 22.5]])
    os_mock = mocker.patch('gpu_tracker.tracker.os', wraps=os, getpid=mocker.MagicMock(return_value=main_process_id))
    time_mock = mocker.patch(
        'gpu_tracker.tracker.time', time=mocker.MagicMock(side_effect=[800, 900, 1000, 1100, 0]), sleep=mocker.MagicMock())
    log_spy = mocker.spy(gput.tracker.log, 'warning')
    sleep_time = 1.5
    join_timeout = 5.5
    if use_context_manager:
        with gput.Tracker(
                sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit, gpu_ram_unit=gpu_ram_unit,
                time_unit=time_unit, gpu_uuids=gpu_uuids, n_expected_cores=n_expected_cores) as tracker:
            pass
    else:
        tracker = gput.Tracker(
            sleep_time=sleep_time, join_timeout=join_timeout, ram_unit=ram_unit, gpu_ram_unit=gpu_ram_unit, time_unit=time_unit,
            gpu_uuids=gpu_uuids, n_expected_cores=n_expected_cores)
        tracker.start()
        tracker.stop()
    assert start_mock.called
    assert not os.path.isfile(tracker._resource_usage_file)
    assert not log_spy.called
    utils.assert_args_list(virtual_memory_mock, [()] * 4)
    system_mock.assert_called_once_with()
    EventMock.assert_called_once_with()
    utils.assert_args_list(mock=tracker._stop_event.is_set, expected_args_list=[()] * 4)
    utils.assert_args_list(mock=PsProcessMock, expected_args_list=[(main_process_id,)] * 2)
    utils.assert_args_list(current_process_mock.children, [()] * 2)
    utils.assert_args_list(mock=main_process_mock.children, expected_args_list=[{'recursive': True}] * 3, use_kwargs=True)
    if operating_system == 'Linux':
        utils.assert_args_list(mock=main_process_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
        utils.assert_args_list(mock=child1_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
        utils.assert_args_list(mock=child2_mock.memory_maps, expected_args_list=[{'grouped': False}] * 6, use_kwargs=True)
    else:
        utils.assert_args_list(mock=main_process_mock.memory_info, expected_args_list=[()] * 6)
        utils.assert_args_list(mock=child1_mock.memory_info, expected_args_list=[()] * 6)
        utils.assert_args_list(mock=child2_mock.memory_info, expected_args_list=[()] * 6)
    assert len(check_output_mock.call_args_list) == 8
    os_mock.getpid.assert_called_once_with()
    utils.assert_args_list(mock=time_mock.time, expected_args_list=[()] * 5)
    cpu_percent_interval = gput.tracker._TrackingProcess._CPU_PERCENT_INTERVAL
    true_sleep_time = sleep_time - cpu_percent_interval
    utils.assert_args_list(
        mock=time_mock.sleep, expected_args_list=[(cpu_percent_interval,), (true_sleep_time,)] * 3)
    tracker._stop_event.set.assert_called_once_with()
    tracker._tracking_process.join.assert_called_once_with(timeout=join_timeout)
    utils.assert_args_list(mock=tracker._tracking_process.is_alive, expected_args_list=[()] * 2)
    assert not tracker._tracking_process.terminate.called
    tracker._tracking_process.close.assert_called_once_with()
    cpu_count_mock.assert_called_once_with()
    utils.assert_args_list(cpu_percent_mock, [()] * 3)
    expected_measurements_file = f'tests/data/{use_context_manager}-{operating_system}-{ram_unit}-{gpu_ram_unit}-{time_unit}'
    with open(f'{expected_measurements_file}.txt', 'r') as file:
        expected_tracker_str = file.read()
        assert expected_tracker_str == str(tracker)
    with open(f'{expected_measurements_file}.json', 'r') as file:
        expected_measurements = json.load(file)
        assert expected_measurements == tracker.to_json()


def test_main_process_warnings(mocker, caplog):
    n_join_attempts = 3
    join_timeout = 5.2
    subprocess_mock = mocker.patch('gpu_tracker.tracker.subp', check_output=mocker.MagicMock(side_effect=FileNotFoundError))
    mocker.patch('gpu_tracker.tracker.time', time=mocker.MagicMock(return_value=23.0))
    mocker.patch('gpu_tracker.tracker.os.path.getmtime', return_value=12.0)
    mocker.patch.object(gput.tracker._TrackingProcess, 'is_alive', return_value=True)
    join_spy = mocker.spy(gput.tracker._TrackingProcess, 'join')
    terminate_spy = mocker.spy(gput.tracker._TrackingProcess, 'terminate')
    close_spy = mocker.spy(gput.tracker._TrackingProcess, 'close')
    with gput.Tracker(n_join_attempts=n_join_attempts, join_timeout=join_timeout) as tracker:
        set_spy = mocker.spy(tracker._stop_event, 'set')
    subprocess_mock.check_output.assert_called_once()
    utils.assert_args_list(mock=set_spy, expected_args_list=[()] * n_join_attempts)
    utils.assert_args_list(
        mock=join_spy, expected_args_list=[{'timeout': join_timeout}] * n_join_attempts, use_kwargs=True)
    utils.assert_args_list(mock=tracker._tracking_process.is_alive, expected_args_list=[()] * (n_join_attempts + 1))
    terminate_spy.assert_called_once()
    close_spy.assert_called_once()
    expected_warnings = [nvidia_smi_unavailable_message]
    expected_warnings += ['The tracking process is still alive after join timout. Attempting to join again...'] * n_join_attempts
    expected_warnings.append(
        'The tracking process is still alive after 3 attempts to join. Terminating the process by force...')
    expected_warnings.append(
        'Tracking is stopping and it has been 11.0 seconds since the temporary tracking results file was last updated. '
        'Resource usage was not updated during that time.')
    assert not os.path.isfile(tracker._resource_usage_file)
    _assert_warnings(caplog, expected_warnings)


def _assert_warnings(caplog, expected_warnings: list[str]):
    for expected_warning, record in zip(expected_warnings, caplog.records):
        assert record.levelname == 'WARNING'
        assert record.message == expected_warning


@pt.fixture(name='disable_logs', params=[True, False])
def get_disable_logs(request) -> bool:
    yield request.param


def test_tracking_process_warnings(mocker, disable_logs: bool, caplog):
    main_process_id = 666
    child_process_id = 777
    error_message = 'Unexpected error'
    ProcessMock = mocker.patch(
        'gpu_tracker.tracker.psutil.Process',
        side_effect=[
            mocker.MagicMock(), psutil.NoSuchProcess(pid=666), mocker.MagicMock(),
            mocker.MagicMock(children=mocker.MagicMock(
                side_effect=[psutil.NoSuchProcess(child_process_id), RuntimeError(error_message)]))])
    subprocess_mock = mocker.patch('gpu_tracker.tracker.subp', check_output=mocker.MagicMock(side_effect=FileNotFoundError))
    log_spy = mocker.spy(gput.tracker.log, 'warning')
    tracker = gput.Tracker(process_id=main_process_id, disable_logs=disable_logs)
    tracker._tracking_process.run()
    os.remove(tracker._resource_usage_file)
    mocker.patch(
        'gpu_tracker.tracker.mproc.Event', return_value=mocker.MagicMock(is_set=mocker.MagicMock(side_effect=[False, False, True])))
    print_mock = mocker.patch('builtins.print')
    tracker = gput.Tracker(process_id=main_process_id, disable_logs=disable_logs)
    tracker._tracking_process.run()
    os.remove(tracker._resource_usage_file)
    utils.assert_args_list(ProcessMock, [(os.getpid(),), (main_process_id,), (os.getpid(),), (main_process_id,)])
    [printed] = print_mock.call_args_list
    [printed] = printed.args
    assert error_message == str(printed)
    assert len(subprocess_mock.check_output.call_args_list) == 2
    if disable_logs:
        assert not log_spy.called
    else:
        expected_warnings = [
            nvidia_smi_unavailable_message, 'The target process of ID 666 ended before tracking could begin.', nvidia_smi_unavailable_message,
            'Failed to track a process (PID: 777) that does not exist. This possibly resulted from the process completing before it could be tracked.',
            'The following uncaught exception occurred in the tracking process:']
        _assert_warnings(caplog, expected_warnings)


def test_validate_arguments(mocker):
    with pt.raises(ValueError) as error:
        gput.Tracker(sleep_time=0.0)
    assert str(error.value) == 'Sleep time of 0.0 is invalid. Must be at least 0.1 seconds.'
    with pt.raises(ValueError) as error:
        gput.Tracker(ram_unit='milibytes')
    assert str(error.value) == '"milibytes" is not a valid RAM unit. Valid values are bytes, gigabytes, kilobytes, megabytes, terabytes'
    subprocess_mock = mocker.patch(
        'gpu_tracker.tracker.subp', check_output=mocker.MagicMock(
            side_effect=[b'', b'uuid ,memory.total [MiB] \ngpu-id1,2048 MiB\ngpu-id2,2048 MiB', b'', b'uuid ,memory.total [MiB] ']))
    with pt.raises(ValueError) as error:
        gput.Tracker(gpu_uuids={'invalid-id'})
    assert len(subprocess_mock.check_output.call_args_list) == 2
    assert str(error.value) == 'GPU UUID of invalid-id is not valid. Available UUIDs are: gpu-id1, gpu-id2'
    with pt.raises(ValueError) as error:
        gput.Tracker(gpu_uuids=set[str]())
    assert len(subprocess_mock.check_output.call_args_list) == 4
    assert str(error.value) == 'gpu_uuids is not None but the set is empty. Please provide a set of at least one GPU UUID.'


def test_state(mocker):
    mocker.patch('gpu_tracker.tracker.subp.check_output', side_effect=FileNotFoundError)
    tracker = gput.Tracker()
    assert tracker.__repr__() == 'State: NEW'
    with pt.raises(RuntimeError) as error:
        tracker.stop()
    assert str(error.value) == 'Cannot stop tracking when tracking has not started yet.'
    tracker.start()
    assert tracker.__repr__() == 'State: STARTED'
    with pt.raises(RuntimeError) as error:
        tracker.start()
    assert str(error.value) == 'Cannot start tracking when tracking has already started.'
    tracker.stop()
    assert tracker.__repr__() == 'State: STOPPED'
    with pt.raises(RuntimeError) as error:
        tracker.start()
    assert str(error.value) == 'Cannot start tracking when tracking has already stopped.'
    with pt.raises(RuntimeError) as error:
        tracker.stop()
    assert str(error.value) == 'Cannot stop tracking when tracking has already stopped.'
