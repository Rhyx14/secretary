from typing import Any
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from functools import partial
def get_name(obj:torch.nn.Module | DistributedDataParallel | partial | Any) -> str:
    '''
    Get the name str of an object.

    if the object is the instance of torch.nn.parallel.distributed.DistributedDataParallel, return the name of object.module

    if the object does not has the `name' property, return object.__name__
    '''
    if isinstance(obj,DistributedDataParallel):
        obj=obj.module
    elif isinstance(obj,partial):
        obj=obj.func
    if hasattr(obj,'name'):
        return obj.name
    else:
        return obj.__name__