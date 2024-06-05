def assert_args_list(mock, expected_args_list: list[tuple | dict], use_kwargs: bool = False):
    actual_args_list = [call.kwargs if use_kwargs else call.args for call in mock.call_args_list]
    assert actual_args_list == expected_args_list
