import os
import torch
class Configuration():
    def __init__(self) -> None:
        pass
    def update(self,params:dict):
        self.__dict__.update(params)
        return self
    
    def load_weight(self,net:torch.nn.Module,strict=False):
        if(hasattr(self,'PRE_TRAIN')):
            net.load_state_dict(torch.load(self.PRE_TRAIN,map_location='cpu'),strict=strict)
    
    def __str__(self) -> str:
        ls=''
        for k,v in self.__dict__.items():
            if str.startswith(k,'__'):
                continue
            if isinstance(v,(str,int,float)):
                ls += "%s\t%s\n" % (k,v)
            else:
                ls += "%s\t%s\n" % (k,v)
        return ls
