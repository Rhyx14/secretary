import torch
import os

class PipelineBase():
    def __init__(self,logger,net,before_hooks,after_hooks) -> None:
        self.logger=logger
        self.net=net
        assert isinstance(self.net,torch.nn.Module)
        self.before_hooks=before_hooks
        self.after_hooks=after_hooks
        pass
    
    def Run(self,*args,**kwds):
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        PipelineBase.call_hooks(self.before_hooks,self)
        rslt=self.Run(*args,**kwds)
        PipelineBase.call_hooks(self.after_hooks,self)
        return rslt
    
    def Eval(self):
        self.net.eval()
    
    def Train(self):
        self.net.train()

    @staticmethod
    def call_hooks(hooks,*args):
        if(hooks is not None):
            if isinstance(hooks,list):
                for h in hooks:
                    h(*args)
            else:
                hooks(*args)

    def _Check_Attribute(self,key:str,type:tuple):
        if hasattr(self,key):
            assert isinstance(self.key,type)
        else:
            raise Exception(f'Missing "{key}" with type "{type}"')