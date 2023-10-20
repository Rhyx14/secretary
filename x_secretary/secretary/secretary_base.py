'''
负责程序输出与输入（控制台，权重文件，日志文件）
'''
import logging
from pathlib import Path
import datetime
import os
import json
import torch.distributed as dist
from ..utils.sys_info import get_host_name
from .solo_method import solo_method,solo_method_with_default_return,solo_chaining_method
from ..data_recorder import data_recorder
class Secretary_base():

    def __init__(self,saved_dir:Path,distributed) -> None:
        
        # default objects
        self.logger=logging.getLogger("secretary")
        self.time_stamps=None
        self.data_recorder=data_recorder()
        self.stages_list=[]
        
        self.distributed=distributed
        if(self.distributed):
            self.LOCAL_RANK=dist.get_rank()        
        else:
            self.LOCAL_RANK=0
        
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

    @solo_method
    def record_serial_data(self,name,index,value):
        '''
        record data (solo)

        name: key of data

        index: index of the current datum

        value: value of the current datum
        '''
        self.data_recorder.record_serial_data(name,index,value)
    
    def get_data(self,name):
        '''
        get data
        '''
        return self.data_recorder.data.get(name,0)

    @solo_method
    def record_data(self,name,value):
        self.data_recorder.record_data(name,value)
    
    @solo_method
    def record_moving_avg(self,name,value,steps):
        '''
        record avg value (solo)

        name: key of data

        value: value of the current datum

        steps: index for calculating average 
        '''
        prev=self.data_recorder.data[name]
        tmp=(prev*steps+ value)/(steps+1)
        self.data_recorder.record_data(name,tmp)
        return tmp

    @solo_chaining_method
    def dump_json(self,filename,obj):
        '''
        dump to json file (solo)
        '''
        with open(self.SAVED_DIR/filename,'w') as f:
            json.dump(obj,f)
        return self

    def sync(self):
        if(self.distributed):
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
        save recorded data (solo) -> self
        '''
        self.data_recorder.save(self.SAVED_DIR)
        return self

    @solo_method
    def info_wechat_autodl(self,token,title,name=None,content=None):    
        # python脚本示例
        import requests
        if(name is None):
            name=get_host_name()
        if content is None:
            content='no content'
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                     json={
                         "token": token,
                         "title": title,
                         "name": name,
                         "content": content
                     })

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

    def register_stage(self,priority,pre_acts=[],post_acts=[],name='default'):
        '''
        添加执行阶段,根据prioirity顺序(由小到大)执行。
        after_actions: 该阶段完成后执行的动作
        '''
        def outter(f,*args,**kwargs):
            self.stages_list.append([
                priority,
                name,
                [*pre_acts,f,*post_acts]
            ])
            def inner():
                f(*args,**kwargs)
            return inner
        return outter
    
    def run_stages(self):
        '''
        执行已注册的训练阶段
        '''
        self.stages_list.sort(key=lambda x: x[0])
        for _,_,stages in self.stages_list:
            for func in stages:
                func()
