from typing import Any, List

def _to_list_if_str(list_):
    return _to_list_if_type(list_, str)

def _to_list_if_tuple(list_):
    return _to_list_if_type(list_, tuple)

def _to_list_if_type(list_, type):
    if isinstance(list_, type):
        list_ = [list_]
    assert isinstance(list_, list)
    return list_

def _to_list_if_not(item: Any) -> List[Any]:
    if not isinstance(item, list):
        return [item]
    return item