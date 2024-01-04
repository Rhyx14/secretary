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
from ..data_recorder import data_recorder
from ..deprecated import deprecated
class Secretary_base():

    def __init__(self,saved_dir:Path) -> None:
        
        # default objects
        self.logger=logging.getLogger("secretary")
        self.time_stamps=None
        self._data=data_recorder()
        self._stages_list=[]
        
        if(dist.is_torchelastic_launched()):
            self.LOCAL_RANK=dist.get_rank()
            self._distributed=True       
        else:
            self.LOCAL_RANK=0
            self._distributed=False
        
        self.SAVED_DIR=saved_dir


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
        self.logger.debug(msg)

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
        self.logger.info(msg)
        return self

    def info_all(self,msg):
        '''
        log information (tutti) -> self
        '''
        self.logger.info(msg)
        return self
    
    @solo_chaining_method
    def warning(self,msg):
        '''
        log warning (solo) -> self
        '''
        self.logger.warning(msg)
        return self

    def warning_all(self,msg):
        '''
        log warning (tutti) -> self
        '''
        self.logger.warning(msg)
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
        with open(self.SAVED_DIR/filename,'w') as f:
            json.dump(obj,f)
        return self

    def sync(self):
        if(self._distributed):
            dist.barrier()
    
    def load_json(self,filename):
        '''
        load from json file (tutti)
        '''
        return json.load(open(self.SAVED_DIR/filename,'r'))

    def exist(self,filename):
        '''
        check file existence (tutti)
        '''
        return os.path.exists(self.SAVED_DIR/filename)
    
    @solo_chaining_method
    def save_record(self):
        '''
        save recorded data in self._data_recorder (solo) -> self
        '''
        self._data.save(self.SAVED_DIR)
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
        if self.time_stamps is None:
            self.time_stamps=datetime.datetime.now()
        else:
            now=datetime.datetime.now()
            span=now-self.time_stamps
            self.logger.info(f'span === {str(span)}')
            self.time_stamps=now
        return self

    def register_stage(self,priority=-1,pre_acts=[],post_acts=[],env_obj=None):
        '''
        添加执行阶段,根据prioirity顺序(由小到大)执行。
        after_actions: 该阶段完成后执行的动作
        '''
        def outter(f,*args,**kwargs):
            
            # 未指定优先级则自动添加（最低优先级）
            if priority == -1:
                priority=len(self._stages_list)+1

            self._stages_list.append([
                priority,
                [*pre_acts,f,*post_acts],
                env_obj
            ])
            def inner():
                f(*args,**kwargs)
            return inner
        return outter
    
    def run_stages(self):
        '''
        执行已注册的训练阶段
        '''
        self._stages_list.sort(key=lambda x: x[0])
        self.stage_env=
        for _,stages,_stages_env in self._stages_list:
            for func in stages:
                self.stage_env=_stages_env
                func()