import os
from ..secretary_solo_method import solo_method
import logging
import torch.distributed as dist
import torch

from ...utils.info import get_sys_info
from ...configuration import Configuration
from .init_base import init_base
from ...utils.log_dir import Log_dir

class Train_init(init_base):
    def __init__(self,cfg, distributed=False,base_folder=False,name_prefix='',logging_level=logging.INFO) -> None:
        assert isinstance(cfg,Configuration)
        self.cfg=cfg
        self.distributed=distributed

        if(self.distributed):
            self.LOCAL_RANK=dist.get_rank()        
        else:
            self.LOCAL_RANK=0

        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.logger=logging.getLogger(cfg.NAME)
        
        # create folder
        self.Log_dir=Log_dir(
            Log_dir.time_suffix_name(name_prefix),
            root_path=base_folder,
            distributed=self.distributed
        ).create_dir()
        self.SAVED_DIR=self.Log_dir.saved_dir
        self.sync()

        # 保存日志
        self.logger.setLevel(logging_level)
        fh=logging.FileHandler(self.SAVED_DIR/'log.txt')
        fh.setFormatter(logging.Formatter('%(asctime)s-[%(name)s] %(message)s'))
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

        self._print_configuration()

        # 将init过程中更新的变量写入secretary的成员中
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
    def _print_configuration(self):
        '''
        ouput the Configuration object

        saving to 'configuration.txt' 
        '''
        cfg_str=str(self.cfg)
        print(cfg_str)
        with open(os.path.join(self.SAVED_DIR,'configuration.txt'),'w') as f:

            # 打印环境信息
            f.write(get_sys_info())
            f.write(cfg_str)
            