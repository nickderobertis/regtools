

def _to_list_if_str(list_):
    if isinstance(list_, str):
        list_ = [list_]
    assert isinstance(list_, list)
    return list_