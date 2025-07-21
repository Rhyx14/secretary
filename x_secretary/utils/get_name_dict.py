from functools import partial
from typing import Any
def get_name_dict(*lists) -> dict:
    '''
    return a dict wherein each key is the name of the object, value the object value
    '''
    _rslt={}
    for list in lists:
        _tmp={_get_name(var): var for var in list}
        _rslt |= _tmp
    return _rslt

def _get_name(obj:partial | Any) -> str:
    '''
    Get the name str of an object.
    '''
    if isinstance(obj,partial):
        obj=obj.func
    else:
        return obj.__name__