import torch
import os
from typing import Any
from contextlib import contextmanager

from tqdm import tqdm
import torch.distributed as dist
def DDP_progressbar(iterator):
    if dist.is_torchelastic_launched() and dist.get_rank()!=0:
        return iterator
    else:
        return tqdm(iterator,leave=False)
    

class PipelineBase():
    def __init__(self,default_device='cpu') -> None:
        self.default_device=default_device
        pass
    
    def _unpack_seg(self,datum):
        return datum['X'].to(self.default_device),datum['Y'].to(self.default_device)

    def _unpack_cls(self,datum):
        return datum[0].to(self.default_device),datum[1].to(self.default_device)
    
    def _unpack(self,datum):
        '''
        unpack data out from the dataloader, default setting is for classification task (return (x, label), then to the device) 
        '''
        return self._unpack_cls(datum)
    
    def Run(self,*args,**kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.Run(*args,**kwds)

    @staticmethod
    def call_hooks(hooks,*args):
        if(hooks is not None):
            if isinstance(hooks,list):
                for h in hooks:
                    h(*args)
            else:
                hooks(*args)

    @staticmethod
    def _Check_Attribute(obj,key:str,type: tuple | Any =None):
        if hasattr(obj,key):
            if type is not None:
                if not isinstance(obj.__dict__[key],type):
                    raise TypeError(f'Attribute "{str(key)}" is not the type "{str(type)}"')
        else:
            raise Exception(f'Missing "{str(key)}" with type "{str(type)}"')
        
    @staticmethod
    @contextmanager
    def switch_eval_train(net):
        net.eval()
        yield
        net.train()