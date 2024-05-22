import gpu_tracker.__main__ as cli
import pytest as pt

test_data = [['-e', ]]


@pt.mark.parametrize('argv', test_data)
def test_main(mocker, argv: list[str]):
    argv = ['gpu-tracker'] + argv
    mocker.patch('sys.argv', argv)
    mocker.patch('gpu_tracker.__main__.subp', )
    cli.main()
