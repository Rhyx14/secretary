def get_name_dict(*lists):
    '''
    return a dict, each key is the name of the object in the list, value the object value
    '''
    _rslt={}
    for list in lists:
        _tmp={var.__name__: var for var in list}
        _rslt |= _tmp
    return _rslt