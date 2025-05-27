from typing import Any
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
def get_name(obj:torch.nn.Module | DistributedDataParallel | Any) -> str:
    '''
    Get the name str of an object.

    if the object is the instance of torch.nn.parallel.distributed.DistributedDataParallel, return the name of object.module

    if the object does not has the `name' property, return object.__name__
    '''
    if isinstance(obj,DistributedDataParallel):
        obj=obj.module
    if hasattr(obj,'name'):
        return obj.name
    else:
        return obj.__name__