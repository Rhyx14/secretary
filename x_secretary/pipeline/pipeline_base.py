import torch
import os
from typing import Any
from contextlib import contextmanager
class PipelineBase():
    def __init__(self,default_device='cpu') -> None:
        self.default_device=default_device
        pass
    
    def _unpack_seg(self,datum):
        return datum['X'].to(self.default_device),datum['Y'].to(self.default_device)

    def _unpack_cls(self,datum):
        return datum[0].to(self.default_device),datum[1].to(self.default_device)
    
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
                assert isinstance(obj.__dict__[key],type)
        else:
            raise Exception(f'Missing "{str(key)}" with type "{str(type)}"')
        
    @staticmethod
    @contextmanager
    def switch_eval_train(net):
        net.eval()
        yield
        net.train()