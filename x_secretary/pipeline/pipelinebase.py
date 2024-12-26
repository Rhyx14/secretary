import torch
import os
from typing import Any
from contextlib import contextmanager

class Default_DataHook:
    @staticmethod
    def unpack_seg_train(datum,device):
        return datum['X'].to(device),datum['Y'].to(device)
    
    @staticmethod
    def unpack_seg_val(datum,device):    
        inputs = datum['X'].to(device)
        gt=datum['Y'].cpu().numpy()
        return inputs,gt
    
    @staticmethod
    def unpack_cuda(datum):
        return datum[0].cuda(),datum[1].cuda()

class PipelineBase():
    def __init__(self,default_device,data_hooks) -> None:
        self.default_device=default_device
        self.data_hooks=data_hooks
        pass
    
    def Run(self,*args,**kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.Run(*args,**kwds)

    @staticmethod
    def call_actions(actions,*args):
        if(actions is not None):
            if isinstance(actions,list):
                for _act in actions:
                    _act(*args)
            else:
                actions(*args)

    @staticmethod
    def call_hooks(hooks,*args):
        if(hooks is not None):
            if isinstance(hooks,list):
                for _hook in hooks:
                    args=_hook(*args)
            else:
                args=hooks(*args)
        return args

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