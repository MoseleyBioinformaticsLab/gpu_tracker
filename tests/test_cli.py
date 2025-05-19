import gpu_tracker.__main__ as cli
import pytest as pt
import os
import utils


@pt.fixture(name='format_', params=['text', 'json', 'pickle', None])
def get_format(request) -> str | None:
    yield request.param


@pt.fixture(name='output', params=['my-file', None])
def get_output(request) -> str | None:
    yield request.param


execute_test_data = [
    (['-e', 'my-command', '--ru=kilobytes'], ['my-command'], {'disable_logs': False, 'overwrite': False, 'ram_unit': 'kilobytes'}),
    (
        ['--execute', 'my-command arg1 ', '--disable-logs', '--overwrite'], ['my-command', 'arg1'],
        {'disable_logs': True, 'overwrite': True}
    ),
    (
        ['--execute=my-command arg1  arg2', '--st=0.4', '--gb=nvidia', '--tf=track-file.sqlite'], ['my-command', 'arg1', 'arg2'],
        {'disable_logs': False, 'overwrite': False, 'sleep_time': 0.4, 'gpu_brand': 'nvidia', 'tracking_file': 'track-file.sqlite'}
    ),
    (
        ['-e', 'my-command', '--gru=megabytes', '--tu=days', '--gb=amd', '--tf=track-file.csv'], ['my-command'],
        {'disable_logs': False, 'overwrite': False, 'gpu_ram_unit': 'megabytes', 'time_unit': 'days', 'gpu_brand': 'amd', 'tracking_file': 'track-file.csv'}),
    (
        ['-e', 'my-command', '--nec=3', '--guuids=gpu-id1,gpu-id2,gpu-id3', '--gb=amd', '--tconfig=tests/data/tconfig.json'],
        ['my-command'],
        {
            'disable_logs': False, 'overwrite': False, 'n_expected_cores': 3, 'gpu_uuids': {'gpu-id1', 'gpu-id2', 'gpu-id3'},
            'gpu_brand': 'amd', 'sleep_time': 0.5, 'ram_unit': 'megabytes', 'gpu_ram_unit': 'megabytes', 'time_unit': 'seconds'
        }
    ),
    (['-e', 'my-command', '--guuids=gpu-id1'], ['my-command'], {'disable_logs': False, 'overwrite': False, 'gpu_uuids': {'gpu-id1'}})]


class PickleableMock:
    text = 'text'
    json_str = '{\n "key": "json"\n}'

    @staticmethod
    def pickle():
        return b'\x80\x04\x955\x00\x00\x00\x00\x00\x00\x00\x8c\x08test_cli\x94\x8c\x0ePickleableMock\x94\x93\x94)\x81\x94}\x94\x8c\x03key\x94\x8c\x06pickle\x94sb.'

    def __str__(self):
        return PickleableMock.text

    @staticmethod
    def to_json():
        return {'key': 'json'}

    def __getstate__(self):
        return {'key': 'pickle'}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self


@pt.mark.parametrize('argv,command,kwargs', execute_test_data)
def test_execute(mocker, argv: list[str], command: list[str], kwargs: dict, format_: str | None, output: str | None):
    _mock_cli(mocker, format_, output, argv)
    argv = ['gpu-tracker'] + argv
    argv += ['-f', format_] if format_ else []
    argv += ['-o', output] if output else []
    mocker.patch('sys.argv', argv)
    process_mock = mocker.MagicMock(returncode=0, pid=666)
    subprocess_mock = mocker.patch('gpu_tracker.__main__.subp', Popen=mocker.MagicMock(return_value=process_mock))
    tracker_mock = PickleableMock()
    TrackerMock_ = mocker.patch('gpu_tracker.__main__.Tracker', return_value=tracker_mock)
    print_args = [('Resource tracking complete. Process completed with status code: 0',)]
    _test_process_output(mocker, format_, output, print_args)
    TrackerMock_.assert_called_with(process_id=process_mock.pid, **kwargs)
    subprocess_mock.Popen.assert_called_once_with(command)
    process_mock.wait.assert_called_once_with()


error_data = [
    (['-e '], 'Empty command provided.'), (['-e', 'my-command'], 'Command not found: "my-command"'),
    (['-e', 'my-command'], f'The following error occurred when starting the command "my-command":'),
    (['-e', 'my-command', '-f', 'invalid-format'], '"invalid-format" is not a valid format. Valid values are "json" or "text".'),
    (
        ['sub-track', 'compare'],
        'A mapping of tracking session name to file path must be provided either through the -m option or a config file.'
    )
]


@pt.mark.parametrize('argv,error_message', error_data)
def test_errors(mocker, argv: list[str], error_message: str):
    argv = ['gpu-tracker'] + argv
    mocker.patch('sys.argv', argv)
    if 'Command not found' in error_message:
        popen_side_effect = FileNotFoundError
    elif 'The following error occurred' in error_message:
        popen_side_effect = Exception
    else:
        popen_side_effect = mocker.MagicMock()
    mocker.patch('gpu_tracker.__main__.subp.Popen', side_effect=popen_side_effect)
    log_mock = mocker.patch('gpu_tracker.__main__.log', error=mocker.MagicMock())
    mocker.patch('gpu_tracker.__main__.Tracker')
    with pt.raises(SystemExit) as error:
        cli.main()
    assert str(error.value) == '1'
    log_mock.error.assert_called_once_with(error_message)


def _mock_cli(mocker, format_: str | None, output: str | None, argv: list[str]):
    argv = ['gpu-tracker'] + argv
    argv += ['-f', format_] if format_ else []
    argv += ['-o', output] if output else []
    mocker.patch('sys.argv', argv)


def _test_process_output(mocker, format_: str, output: str | None, print_args: list[tuple[str | bytes]] | None = None):
    print_args = list[tuple[str | bytes]]() if print_args is None else print_args
    print_mock = mocker.patch('builtins.print')
    cli.main()
    if format_ == 'text' or format_ is None:
        output_str = PickleableMock.text
    elif format_ == 'json':
        output_str = PickleableMock.json_str
    else:
        output_str = PickleableMock.pickle()
    if output is None:
        print_args.append((output_str,))
    else:
        with open(output, 'r' if format_ != 'pickle' else 'rb') as file:
            assert output_str == file.read()
        os.remove(output)
    utils.assert_args_list(print_mock, print_args)


def test_analyze(mocker, format_: str | None, output: str | None):
    _mock_cli(mocker, format_, output, ['sub-track', 'analyze', '--tf=tracking.sqlite', '--stf=sub-tracking.csv'])
    analyzer_mock = mocker.MagicMock(sub_tracking_results=mocker.MagicMock(return_value=PickleableMock()))
    SubTrackingAnalyzerMock = mocker.patch('gpu_tracker.__main__.SubTrackingAnalyzer', return_value=analyzer_mock)
    _test_process_output(mocker, format_, output)
    SubTrackingAnalyzerMock.assert_called_once_with('tracking.sqlite', 'sub-tracking.csv')
    analyzer_mock.sub_tracking_results.assert_called_once_with()


combine_test_data = [
    (['-p', 'file1.csv', '-p', 'file2.csv', '-p', 'file3.csv'], ['file1.csv', 'file2.csv', 'file3.csv']),
    (
        ['-p', 'tests/data/sub-tracking-results/files-to-combine/'],
        [
            'tests/data/sub-tracking-results/files-to-combine/1723811.sub-tracking.csv',
            'tests/data/sub-tracking-results/files-to-combine/1723811.sub-tracking.sqlite',
            'tests/data/sub-tracking-results/files-to-combine/1723814.sub-tracking.csv',
            'tests/data/sub-tracking-results/files-to-combine/1723814.sub-tracking.sqlite',
            'tests/data/sub-tracking-results/files-to-combine/1723815.sub-tracking.csv',
            'tests/data/sub-tracking-results/files-to-combine/1723815.sub-tracking.sqlite',
            'tests/data/sub-tracking-results/files-to-combine/main.sub-tracking.csv',
            'tests/data/sub-tracking-results/files-to-combine/main.sub-tracking.sqlite'
        ]
    )
]


@pt.mark.parametrize('argv,files', combine_test_data)
def test_combine(mocker, argv, files: list[str]):
    _mock_cli(mocker, None, None, ['sub-track', 'combine', '--stf=sub-tracking.csv'] + argv)
    analyzer_mock = mocker.MagicMock(combine_sub_tracking_files=mocker.MagicMock())
    SubTrackingAnalyzerMock = mocker.patch('gpu_tracker.__main__.SubTrackingAnalyzer', return_value=analyzer_mock)
    cli.main()
    SubTrackingAnalyzerMock.assert_called_once_with(None, 'sub-tracking.csv')
    analyzer_mock.combine_sub_tracking_files.assert_called_once_with(files)


compare_test_data = [
    (['-m', 'A=file1.pkl', '-m', 'B=file2.pkl'], 'mean'),
    (['--cconfig=tests/data/cconfig.json', '--stat=min'], 'min'),
    (['--cconfig=tests/data/cconfig.json'], 'max')
]


@pt.mark.parametrize('argv,statistic', compare_test_data)
def test_compare(mocker, format_: str | None, output: str | None, argv, statistic: str):
    _mock_cli(mocker, format_, output, ['sub-track', 'compare'] + argv)
    comparison_mock = mocker.MagicMock(compare=mocker.MagicMock(return_value=PickleableMock()))
    TrackingComparisonMock = mocker.patch('gpu_tracker.__main__.TrackingComparison', return_value=comparison_mock)
    _test_process_output(mocker, format_, output)
    TrackingComparisonMock.assert_called_once_with({'A': 'file1.pkl', 'B': 'file2.pkl'})
    comparison_mock.compare.assert_called_once_with(statistic)
