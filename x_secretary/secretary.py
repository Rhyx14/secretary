'''
负责程序输出与输入（控制台，权重文件，日志文件）
'''
import torch
import datetime
import shutil
import os
import json
from collections import defaultdict
from .utils.info import get_host_name
from .secretary_solo_method import solo_method
from .data_recorder import data_recorder
class Secretary():

    def __init__(self,init:callable) -> None:
        
        init(self)
        self.avg_loss=0
        self.time_stamps=None

        self.data_recorder=data_recorder()

    @solo_method
    def print_solo(self,str,end='\n'):
        print(str,end=end)

    def debug(self,msg):
        '''
        logger debug (tutti)
        '''
        self.logger.debug(msg)

    @solo_method
    def solo(self,callable,*args):
        '''
        run method (solo)
        '''
        return callable(args)
    
    @solo_method
    def info(self,msg):
        '''
        log information (solo)
        '''
        self.logger.info(msg)

    def info_all(self,msg):
        '''
        log information (tutti)
        '''
        self.logger.info(msg)
    
    @solo_method
    def record_serial_data(self,name,index,value):
        '''
        record data (solo)

        name: key of data

        index: index of the current datum

        value: value of the current datum
        '''
        self.data_recorder.record_serial_data(name,index,value)
    
    @solo_method
    def get_data(self,name):
        '''
        get data (solo)
        '''
        return self.data_recorder.data[name]

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

    @solo_method
    def save(self,net=None):
        '''
        save network weight and recorded data (solo)
        '''
        if net is not None:
            if isinstance(net,(torch.nn.parallel.DataParallel,torch.nn.parallel.distributed.DistributedDataParallel)):
                _net=net.module
            else:
                _net=net
            path=os.path.join(self.SAVED_DIR,'weight.pt')
            torch.save(_net.state_dict(),path)
            self.logger.info(f'saved at {path}')
        self.data_recorder.save(self.SAVED_DIR)

    @solo_method
    def dump_json(self,filename,obj):
        '''
        dump to json file (solo)
        '''
        with open(os.path.join(self.SAVED_DIR,filename),'w') as f:
            json.dump(obj,f)

    def load_json(self,filename):
        '''
        load from json file (tutti)
        '''
        return json.load(open(os.path.join(self.SAVED_DIR,filename),'r'))

    def exist(self,filename):
        '''
        check file existence (tutti)
        '''
        return os.path.exists(os.path.join(self.SAVED_DIR,filename))

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
    
    @solo_method
    def shut_down(self,t=1):
        '''
        定时关机, 默认1min (solo)
        '''
        if(self.distributed):
            import torch.distributed as dist
            dist.barrier()
        os.system(f'shutdown -h {t}')
    
    @solo_method
    def timing(self):
        '''
        计算时间间隔 (solo)
        '''
        if self.time_stamps is None:
            self.time_stamps=datetime.datetime.now()
        else:
            now=datetime.datetime.now()
            span=now-self.time_stamps
            self.logger.info(f'span === {str(span)}')
            self.time_stamps=now