import os,logging,sys,torch,argparse
import torch.nn.parallel.distributed
import json,yaml
import torch.distributed as dist
from typing import Any, Union
import pathlib
import re
from .deprecated import deprecated
class Configuration():
    def __init__(self,init_dict:dict=None,auto_record:bool=True,logger=None) -> None:
        '''
        The configuation object.

        auto_record: record the item changes that its key doesn't started with '_'

        logger: whether using external logger. If logger is set to None, a new logger will be created
        '''
        self._auto_record=auto_record
        self._change_set=set()

        if logger is None:
            self.logger=logging.getLogger('CONFIGURATION')
            logger_handler=logging.StreamHandler(sys.stdout)
            logger_handler.setFormatter(logging.Formatter('%(asctime)s-<%(name)s>[%(levelname)s] %(message)s'))
            self.logger.addHandler(logger_handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate=False
                
        else:
            self.logger=logger

        self._parser=argparse.ArgumentParser()
        self.NAME='default'

        # check whether this process is elastic launched, mostly for ddp training
        self.DDP=dist.is_torchelastic_launched()

        if init_dict is not None:
            self.update(init_dict) 

    def set_logger(self,logger):
        self.logger=logger

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
    
    def update_from_files(self,path:pathlib.Path | list | str):
        if not isinstance(path,list):
            path=[path]
        file_list=map(lambda _p : pathlib.Path(_p), path)
        for _p in file_list:
            match _p.suffix:
                case '.json':
                    self.update(json.loads(_p.read_text('uft-8')))
                case '.yaml':
                    stream=_p.open()
                    self.update(yaml.safe_load(stream))
                    stream.close()
                case _ :
                    raise NotImplementedError
        return self

    def __str__(self) -> str:
        ls=''
        for k,v in self.__dict__.items():
            if str.startswith(k,'_'):
                continue
            elif isinstance(v,(torch.Tensor)):
                ls += f"torch.Tensor<shape={v.shape}, dtype={v.dtype}>"
            else:
                ls += "%s\t%s\n" % (k,v)
        return ls

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not str.startswith(__name,'_') and self._auto_record:
            self._change_set.add(__name)
        self.__dict__[__name]=__value
        pass

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