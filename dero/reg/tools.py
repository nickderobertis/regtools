

def _to_list_if_str(list_):
    return _to_list_if_type(list_, str)

def _to_list_if_tuple(list_):
    return _to_list_if_type(list_, tuple)

def _to_list_if_type(list_, type):
    if isinstance(list_, type):
        list_ = [list_]
    assert isinstance(list_, list)
    return list_