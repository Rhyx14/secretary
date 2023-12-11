import random
def random_function(func):
    def inner(*data,p=0.5,**kwds):
        if random.random()<p:
            return func(*data,p=p,**kwds)
        else:
            return data
    return inner
