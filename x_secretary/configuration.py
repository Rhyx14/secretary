import os
import torch
import argparse
from typing import Any, Union
class Configuration():
    def __init__(self,auto_record=False) -> None:
        self._auto_record=auto_record
        self._change_list=[]
        
        self._parser=argparse.ArgumentParser()
        self.NAME='default'

        pass

    def update(self,params:dict):
        for k,v in params.items():
            self.__setattr__(k,v)
        return self
    
    def load_weight(self,net:torch.nn.Module,strict=False,weight_key='PRE_TRAIN'):
        if(hasattr(self,weight_key)):
            net.load_state_dict(torch.load(self.__dict__[weight_key],map_location='cpu'),strict=strict)
        else:
            print(f'no such weight file: {weight_key}')
    
    def __str__(self) -> str:
        ls=''
        for k,v in self.__dict__.items():
            if str.startswith(k,'_'):
                continue
            if isinstance(v,(str,int,float)):
                ls += "%s\t%s\n" % (k,v)
            else:
                ls += "%s\t%s\n" % (k,v)
        return ls

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not str.startswith(__name,'_') and self._auto_record:
            self._change_list.append(__name)
        self.__dict__[__name]=__value
        pass

    def reset_records(self):
        '''
        clear records of changed property name
        '''
        self._change_list.clear()
        return self

    def get_records_str(self):
        '''
        stringify the changed properties
        '''
        ls=''
        for _name in self._change_list:
            v=self.__dict__[_name]
            if isinstance(v,(str,int,float)):
                ls += "%s\t%s\n" % (_name,v)
            else:
                ls += "%s\t%s\n" % (_name,v)
        return ls
    
    def add_args(self,args:Union[list,tuple]):
        '''
        arg_list: (name,type,help_info) 
                or [(name1,type,help_info),(name2,type,help_info),]
        '''
        if isinstance(args,list):
            for name,type,help in args:
                self._parser.add_argument(name,help=help,type=type)
        else:
            self._parser.add_argument(args[0],help=args[1],type=args[2])

        _dist=self._parser.parse_args()
        self.update(_dist.__dict__)
        return self