import gpu_tracker.__main__ as cli
import pytest as pt
import os
import utils


@pt.fixture(name='format_', params=['text', 'json', None])
def get_format(request) -> str:
    yield request.param


@pt.fixture(name='output', params=['my-file', None])
def get_output(request) -> str:
    yield request.param


test_data = [
    (['-e', 'my-command', '--ru=kilobytes'], ['my-command'], {'disable_logs': False, 'ram_unit': 'kilobytes'}),
    (['--execute', 'my-command arg1 ', '--disable-logs'], ['my-command', 'arg1'], {'disable_logs': True}),
    (['--execute=my-command arg1  arg2', '--st=0.4'], ['my-command', 'arg1', 'arg2'], {'disable_logs': False, 'sleep_time': 0.4}),
    (
        ['-e', 'my-command', '--gru=megabytes', '--tu=days'], ['my-command'],
        {'disable_logs': False, 'gpu_ram_unit': 'megabytes', 'time_unit': 'days'}
    )]


@pt.mark.parametrize('argv,command,kwargs', test_data)
def test_main(mocker, argv: list[str], command: list[str], kwargs: dict, format_: str | None, output: str | None):
    argv = ['gpu-tracker'] + argv
    argv += ['-f', format_] if format_ else []
    argv += ['-o', output] if output else []
    mocker.patch('sys.argv', argv)
    process_mock = mocker.MagicMock(returncode=0, pid=666)
    subprocess_mock = mocker.patch('gpu_tracker.__main__.subp', Popen=mocker.MagicMock(return_value=process_mock))
    tracker_str = 'tracker-str'
    tracker_json = {'tracker': 'json'}
    tracker_mock = mocker.MagicMock(
        __str__=mocker.MagicMock(return_value=tracker_str), to_json=mocker.MagicMock(return_value=tracker_json), __enter__=lambda self: self)
    TrackerMock = mocker.patch('gpu_tracker.__main__.Tracker', return_value=tracker_mock)
    print_mock = mocker.patch('builtins.print')
    cli.main()
    TrackerMock.assert_called_with(process_id=process_mock.pid, **kwargs)
    subprocess_mock.Popen.assert_called_once_with(command)
    process_mock.wait.assert_called_once_with()
    if format_ == 'text' or format_ is None:
        tracker_mock.__str__.assert_called_once_with()
        output_str = tracker_str
    else:
        tracker_mock.to_json.assert_called_once_with()
        output_str = '{\n "tracker": "json"\n}'
    print_args = [('Resource tracking complete. Process completed with status code: 0',)]
    if output is None:
        print_args.append((output_str,))
    else:
        with open(output, 'r') as file:
            assert output_str == file.read()
        os.remove(output)
    utils.assert_args_list(print_mock, print_args)
