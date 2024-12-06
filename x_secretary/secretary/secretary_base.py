'''
负责程序输出与输入（控制台，权重文件，日志文件）
'''
import logging
from pathlib import Path
import datetime
import os
import json
import torch.distributed as dist
import torch
from ..utils.autodl import info_wechat_autodl
from .solo_method import solo_method,solo_method_with_default_return,solo_chaining_method
from ..utils.data_recorder import data_recorder
from ..deprecated import deprecated
from functools import wraps

class Secretary_base():

    def __init__(self,working_dir:Path,logger_name:str='secretary',logging_level=logging.INFO) -> None:
        
        # default objects
        self._logger=logging.getLogger(logger_name)
        self._logger.propagate=False
        self._logging_level=logging_level
        self._logger.setLevel(logging_level)

        self._default_logging_formatter=logging.Formatter('%(asctime)s-<%(name)s>[%(levelname)s] %(message)s')
        _ch=logging.StreamHandler()
        _ch.setLevel(self._logging_level)
        _ch.setFormatter(self._default_logging_formatter)
        self._logger.addHandler(_ch)

        self._time_stamps=None
        self._data=data_recorder()

        self._stages_list=[]
        self._stage_act_template=['todo']
        self.stage_env=None
        
        if(dist.is_torchelastic_launched()):
            self.LOCAL_RANK=dist.get_rank()
            self._distributed=True       
        else:
            self.LOCAL_RANK=0
            self._distributed=False
        
        self._working_dir=working_dir

    @property
    def logger(self):
        return self._logger

    @solo_chaining_method
    def print_solo(self,str,**kwargs):
        '''
        print function wrapper (solo) -> self 
        '''
        print(str,**kwargs)
        return self

    def debug(self,msg):
        '''
        logger debug (tutti)
        '''
        self._logger.debug(msg)

    @solo_method
    def solo(self,callable,*args,**kwargs):
        '''
        run method (solo)
        '''
        return callable(*args,**kwargs)
    
    @solo_chaining_method
    def info(self,msg):
        '''
        log information (solo) -> self
        '''
        self._logger.info(msg)
        return self

    def info_all(self,msg):
        '''
        log information (tutti) -> self
        '''
        self._logger.info(msg)
        return self
    
    @solo_chaining_method
    def warning(self,msg):
        '''
        log warning (solo) -> self
        '''
        self._logger.warning(msg)
        return self

    def warning_all(self,msg):
        '''
        log warning (tutti) -> self
        '''
        self._logger.warning(msg)
        return self
# ------------------------------- data record -------------------------------
    @property
    def data(self):
        return self._data
    
# ----------------------------------------------------------------------------
    @solo_chaining_method
    def dump_json(self,filename,obj):
        '''
        dump to json file (solo)
        '''
        with open(self._working_dir/filename,'w') as f:
            json.dump(obj,f)
        return self
    
    def load_json(self,filename):
        '''
        load from json file (tutti)
        '''
        return json.load(open(self._working_dir/filename,'r'))

    def sync(self):
        if(self._distributed):
            dist.barrier()

    def load_weight(self,net:torch.nn.Module,weight=str | dict | Path, strict=False,include:list=None,exclude:list=None):
        """Load weight of the module.

        load the weight from the path.

        'include' means only specific layers are loaded.

        'exclude' means layers other than designated layers will be loaded.

        Args:
            net (torch.nn.Module): torch module. if it is DataParallel or DistributedDataParallel, net.module will be selected automatically.
            weight (str | dict | Path): weight path. Defaults to None.
            strict (bool, optional): strict mode, same as the arg in torch.load. Defaults to False.
            include (list, optional): include pattern (re). Defaults to None.
            exclude (list, optional): exlude pattern (re). Defaults to None.

        Returns:
            None
        """
        if isinstance(net,(torch.nn.DataParallel,torch.nn.parallel.distributed.DistributedDataParallel)):
            net=net.module

        if isinstance(weight,(str,Path)):
            self.logger.info(f"Loading weight from {weight}, including: {include}, excluding: {exclude}")
            weight=torch.load(weight,map_location='cpu')
        else:
            self.logger.info(f"Loading weight, including: {include}, excluding: {exclude}")

        if include is not None:
            if exclude != None or strict !=False:
                raise ValueError(f'"strict = True" and exclude list are not comaptible with "include".') 
            tmp={}
            for _key,_value in weight.items():
                for _in_pattern in include:
                    if re.match(_in_pattern,_key) is not None:
                        tmp[_key]=_value
            weight=tmp
            
        if exclude is not None:
            if include != None or strict !=False:
                raise ValueError(f'"strict = True" and include list are not comaptible with "exclude".') 
            for _ex_pattern in exclude:
                _keys=list(weight.keys())
                for _key in _keys:
                    if re.match(_ex_pattern,_key) is not None:
                        del weight[_key]
        
        net.load_state_dict(weight,strict=strict)
        return self

    def exist(self,filename):
        '''
        check file existence (tutti)
        '''
        return os.path.exists(self._working_dir/filename)
    
    @solo_chaining_method
    def save_record(self):
        '''
        save recorded data in self._data_recorder (solo) -> self
        '''
        self._data.save(self._working_dir)
        return self

    @solo_method
    def info_wechat_autodl(self,token,title,name=None,content=None):    
        info_wechat_autodl(token,title,name,content)


    _divider={
        'KB':1024,
        'MB':1024*1024,
        'GB':1024*1024*1024,
        'TB':1024*1024*1024*1024
    }
    def cuda_VRAM_usage(self,mode='MB'):
        self.info_all(f'{self.LOCAL_RANK} - maximal CUDA memory usage:{torch.cuda.max_memory_allocated()/Secretary_base._divider[mode]:.3f}{mode}')
        return self

    @solo_chaining_method
    def timing(self):
        '''
        计算时间间隔 (solo) -> self
        '''
        if self._time_stamps is None:
            self._time_stamps=datetime.datetime.now()
        else:
            now=datetime.datetime.now()
            span=now-self._time_stamps
            self._logger.info(f'span === {str(span)}')
            self._time_stamps=now
        return self

    def register_stage(self,priority=-1,pre_acts=[],post_acts=[],stage_env=None):
        '''
        添加执行阶段

        ---
        prioirity: 根据顺序(由小到大)执行

        pre_acts: 该阶段完成前执行的动作

        post_acts: 该阶段完成后执行的动作

        env_obj: 一些环境变量,通过secretary实例的stage_env访问
        '''
        
        def outter(f,*args,**kwargs):
            
            # 未指定优先级则自动添加（最低优先级）
            if priority == -1:
                _priority=len(self._stages_list)+1
            else:
                _priority=priority
            
            _act_list=[]
            for _2_act in self._stage_act_template:
                if _2_act !='todo':
                    _act_list.append(_2_act)
                else:
                    _act_list.extend(pre_acts)
                    _act_list.append(f)
                    _act_list.extend(post_acts)
            
            self._stages_list.append((_priority,_act_list,stage_env))
            
            @wraps(f)
            def inner():
                f(*args,**kwargs)
            return inner
        return outter
    
    def set_act_template(self,act_template: list):
        '''
        set the act template 
        ---

        Eg:

        [funcA, funcB, 'todo', funcC, funcD] means, the stage will excute at the sequence as the placeholder 'todo'. 
        '''
        self._stage_act_template=act_template
        if 'todo' not in self._stage_act_template:
            raise ValueError(f'You need to set the placeholder "todo" somewhere')
        
    def run_stages(self):
        '''
        执行已注册的训练阶段
        '''
        self._stages_list.sort(key=lambda x: x[0])
        for _,stages,_stages_env in self._stages_list:
            for func in stages:
                self.stage_env=_stages_env
                func()