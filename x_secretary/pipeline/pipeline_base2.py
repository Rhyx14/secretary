import torch
import os
from contextlib import contextmanager
class PipelineBase2():
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
    def _Check_Attribute(obj,key:str,type:tuple):
        if hasattr(obj,key):
            assert isinstance(obj.__dict__[key],type)
        else:
            raise Exception(f'Missing "{key}" with type "{type}"')
        
    @staticmethod
    @contextmanager
    def switch_eval_train(net):
        net.eval()
        yield
        net.train()