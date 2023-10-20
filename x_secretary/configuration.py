import os
import torch
import argparse
from typing import Any, Union
import re
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
    
    def update_rt(self,name,value):
        '''
        update the configuration object but immediately return
        '''
        self.__setattr__(name,value)
        return value
    
    def load_weight(self,net:torch.nn.Module,strict=False,weight_key='PRE_TRAIN',path=None,include=None,exclude=None):
        _weight:dict=None
        if path is not None:
            _weight=torch.load(path,map_location='cpu')
        elif hasattr(self,weight_key):
            _weight=torch.load(self.__dict__[weight_key],map_location='cpu')
        
        if _weight is not None:
            if include !=None and exclude !=None:
                raise ValueError('param "include" and "exclude" are exclusive')  
            if include !=None or exclude !=None:
                if strict==True : raise ValueError('"strict = True" is not compatible with "include" or "exclude".')  

            if include is not None:
                _tmp={}
                for _key,_value in _weight.items():
                    for _in_pattern in include:
                        if re.match(_in_pattern,_key) is not None:
                            _tmp[_key]=_value
                _weight=_tmp
                
            if exclude is not None:
                for _ex_pattern in exclude:
                    _keys=list(_weight.keys())
                    for _key in _keys:
                        if re.match(_ex_pattern,_key) is not None:
                            del _weight[_key]
            
            net.load_state_dict(_weight,strict=strict)

        else:
            print(f'No desginated path, and such weight file: {weight_key}')
        return self
    
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
    
    def get_records_str_and_reset(self):
        '''
        stringify the changed properties and reset
        '''
        ls=self.get_records_str()
        self.reset_records()
        return ls
    
    def add_args(self,args:Union[list,tuple]):
        '''
        adding arguments in the CMD

        args: (name,type,help_info) or (name,type,default,info) 
                or list of the above tuple.
        '''
        if isinstance(args,list):
            for _tuple in args:
                if len(_tuple)==3:
                    self._parser.add_argument(_tuple[0],help=_tuple[2],type=_tuple[1])
                if len(_tuple)==4:
                    self._parser.add_argument(_tuple[0],help=_tuple[3],type=_tuple[1],default=_tuple[2])
        else:
            if len(args)==3:
                self._parser.add_argument(args[0],help=args[2],type=args[1])
            if len(args)==4:
                self._parser.add_argument(args[0],help=args[3],type=args[1],default=args[2])

        _dist=self._parser.parse_args()
        self.update(_dist.__dict__)
        return self