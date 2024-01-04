import os
import torch
import argparse
import torch.distributed as dist
from typing import Any, Union
import toml
import re
class Configuration():
    def __init__(self,init_dict:dict=None,auto_record:bool=True) -> None:
        self._auto_record=auto_record
        self._change_set=set()
        
        self._parser=argparse.ArgumentParser()
        self.NAME='default'

        # check whether this process is elastic launched, mostly for ddp training
        self.DDP=dist.is_torchelastic_launched()

        if init_dict is not None:
            self.update(init_dict) 

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
        """Load weight of networks.

        If 'path' is given, load the weight from the path.
        
        else if 'weight_key' is given, load the configuration[weight_key].

        'include' means only specific layers are loaded.

        'exclude' means layers other than designated layers will be loaded.

        Args:
            net (torch.nn.Module): the torch module
            strict (bool, optional): strict mode, same as the arg in torch.load. Defaults to False.
            weight_key (str, optional): default key of weight in the object. Defaults to 'PRE_TRAIN'.
            path (_type_, optional): weight path. Defaults to None.
            include (_type_, optional): include pattern (re). Defaults to None.
            exclude (_type_, optional): exlude pattern (re). Defaults to None.

        Returns:
            None
        """
        _weight:dict=None
        if path is not None:
            _weight=torch.load(path,map_location='cpu')
            self.__setattr__(weight_key,str(path))
            
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
            self._change_set.add(__name)
        self.__dict__[__name]=__value
        pass

    def load(self,path):
        '''
        load configurations from a toml file 
        '''
        with open(str(path),'r') as f:
            self.update(toml.load(f))

    def reset_records(self):
        '''
        clear records of changed property name
        '''
        self._change_set.clear()
        return self

    def get_records_str(self):
        '''
        stringify the changed properties
        '''
        ls=''
        for _name in self._change_set:
            v=self.__dict__[_name]
            if isinstance(v,(str,int,float)):
                ls += "%s\t%s\n" % (_name,v)
            else:
                ls += "%s\t%s\n" % (_name,v)
        return ls
    
    def get_records_str_and_reset(self):
        '''
        stringify the changed properties and reset the record
        '''
        ls=self.get_records_str()
        self.reset_records()
        return ls
    
    def spot(self):
        '''
        Alias of get_records_str_and_reset()
        '''
        return self.get_records_str_and_reset()
    
    def _process_args(self,_tuple):
        match len(_tuple):
            case 2: self._parser.add_argument(_tuple[0],action='store_true',help=_tuple[1])
            case 3: self._parser.add_argument(_tuple[0],help=_tuple[2],type=_tuple[1])
            case 4: self._parser.add_argument(_tuple[0],help=_tuple[3],type=_tuple[1],default=_tuple[2])
            case 5: self._parser.add_argument(_tuple[0],_tuple[1],help=_tuple[4],type=_tuple[2],default=_tuple[3])
            case _: raise ValueError(f'Invalid length of tuple: {len(_tuple)}, which should be 2,3,4 or 5')

    def add_args(self,args:Union[list,tuple]):
        '''
        adding arguments in the CMD

        args: 
            (name,type,help_info) 

            or (name,type,default,info) 

            or (short_name,name,type,default,info)
            
            or (flags,help_info)
            
            or list of the above tuple.
        '''
        if isinstance(args,list):
            for _tuple in args:
                self._process_args(_tuple)
        else:
            self._process_args(args)

        _dist=self._parser.parse_args()
        self.update(_dist.__dict__)
        return self