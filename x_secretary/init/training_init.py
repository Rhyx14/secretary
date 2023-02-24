import datetime
import os
import shutil
from ..secretary_solo_method import solo_method
import logging
import torch.distributed as dist
import torch

from ..utils.info import get_sys_info

from .init_base import init_base

class Train_init(init_base):
    def __init__(self,cfg, distributed=False,base_folder=False,name_prefix='',logging_level=logging.INFO) -> None:
        self.cfg=cfg
        self.LOCAL_RANK=0
        self.SAVED_DIR=os.path.join(base_folder,name_prefix+datetime.datetime.now().strftime('_%m%d%H%M')) 
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.logger=logging.getLogger(cfg.NAME)
        self.distributed=distributed

        # sync path
        if(self.distributed):
            self.LOCAL_RANK=cfg.LOCAL_RANK
            ls=[self.SAVED_DIR]
            dist.broadcast_object_list(ls,0)
            self.SAVED_DIR=ls[0] 

        self._init_save_dir()
        self.sync()

        # 保存日志
        self.logger.setLevel(logging_level)
        fh=logging.FileHandler(os.path.join(self.SAVED_DIR,'log.txt'))
        fh.setFormatter(logging.Formatter('%(asctime)s-[%(name)s] %(message)s'))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        self._print_configuration()

        self._updated={
            'cfg':self.cfg,
            'LOCAL_RANK':self.LOCAL_RANK,
            'SAVED_DIR':self.SAVED_DIR,
            'logger':self.logger,
            'distributed':self.distributed
        }
        pass

    def sync(self):
        if(self.distributed):
            dist.barrier()

    @solo_method
    def _init_save_dir(self):
        '''
        初始化文件夹
        '''
        # 原来存在的文件夹先删除
        if os.path.exists(self.SAVED_DIR):
            shutil.rmtree(self.SAVED_DIR,ignore_errors=True)
        os.mkdir(self.SAVED_DIR)
    
    @solo_method
    def _print_configuration(self):
        '''
        打印配置文件
        '''
        cfg_str=str(self.cfg)
        print(cfg_str)
        with open(os.path.join(self.SAVED_DIR,'configuration.txt'),'w') as f:

            # 打印环境信息
            f.write(get_sys_info())
            f.write(cfg_str)
            