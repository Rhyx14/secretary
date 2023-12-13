import random
from functools import wraps
def random_function(func):
    @wraps(func)
    def inner(*data,p=0.5,**kwds):
        if random.random()<p:
            return func(*data,p=p,**kwds)
        else:
            if len(data)==1:
                return data[0]
            else:
                return data
    return inner
