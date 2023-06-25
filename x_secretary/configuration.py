import os
import torch
import argparse
from typing import Union
class Configuration():
    def __init__(self) -> None:
        self.NAME='default'
        self.parser=argparse.ArgumentParser()
        pass

    def update(self,params:dict):
        self.__dict__.update(params)
        return self
    
    def load_weight(self,net:torch.nn.Module,strict=False,weight_key='PRE_TRAIN'):
        if(hasattr(self,weight_key)):
            net.load_state_dict(torch.load(self.__dict__[weight_key],map_location='cpu'),strict=strict)
        else:
            print(f'no such weight file: {weight_key}')
    
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
    
    def add_args(self,args:Union[list,tuple]):
        '''
        arg_list: (name,type,help_info) 
                or [(name1,type,help_info),(name2,type,help_info),]
        '''
        if isinstance(args,list):
            for name,type,help in args:
                self.parser.add_argument(name,help=help,type=type)
        else:
            self.parser.add_argument(args[0],help=args[1],type=args[2])

        _dist=self.parser.parse_args()
        self.update(_dist.__dict__)
        return self