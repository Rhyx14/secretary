from ..solo_method import solo_method
import logging
import torch.distributed as dist
from ...utils.info import get_sys_info
from .init_base import init_base
from ...utils.log_dir import Log_dir

import uuid
class Train_init2(init_base):
    '''
    Train_init2, support the stage running of secretary
    '''
    def __init__(self, distributed=False,base_folder=False,name_prefix='Train_init2',logging_level=logging.INFO) -> None:

        self.distributed=distributed

        if(self.distributed):
            self.LOCAL_RANK=dist.get_rank()        
        else:
            self.LOCAL_RANK=0

        # create folder
        self.Log_dir=Log_dir(
            str(uuid.uuid1()),
            root_path=base_folder,
            distributed=self.distributed
        ).create_dir()
        self.SAVED_DIR=self.Log_dir.saved_dir
        # if(self.distributed):
        #     dist.barrier()

        # 保存日志
        self._logging_level=logging_level
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.logger=logging.getLogger(name_prefix)
        self.logger.setLevel(logging_level)
        self._add_logger_file_handler(self.SAVED_DIR/'log.txt')

        self._print_env()

        # 将init过程中更新的变量写入secretary的成员中
        self._updated={
            'LOCAL_RANK':self.LOCAL_RANK,
            'SAVED_DIR':self.SAVED_DIR,
            'logger':self.logger,
            'distributed':self.distributed,
            'log_to_cfg':self.log_to_cfg,
            'set_name_prefix': self.set_name_prefix,
        }
        pass

    @solo_method
    def _print_env(self):
        '''
        ouput the environment

        saving to 'configuration.txt' 
        '''
        # print(cfg_str)
        with open(self.SAVED_DIR/'configuration.txt','w') as f:
            # 打印环境信息
            f.write(get_sys_info())

    @solo_method
    def log_to_cfg(self,s,prefix=''):
        '''
        log string to the configuration files,

        s should be str or callable object 
        '''
        with open(self.SAVED_DIR/'configuration.txt','a') as f:
            f.write(prefix)
            if not isinstance(s,str):
                s=s()
            f.write(s)
    
    def _add_logger_file_handler(self,path):
        if hasattr(self,'_logger_filehandler'):
            self.logger.removeHandler(self._logger_filehandler)
        self._logger_filehandler=logging.FileHandler(path)
        self._logger_filehandler.setFormatter(logging.Formatter('%(asctime)s-[%(name)s] %(message)s'))
        self._logger_filehandler.setLevel(logging.INFO)
        self.logger.addHandler(self._logger_filehandler)
        pass
        
    def set_name_prefix(self,name_prefix):
        self.Log_dir.change_name(Log_dir.time_suffix_name(name_prefix))
        self.logger.name=name_prefix
        self.SAVED_DIR=self.Log_dir.saved_dir
        self._add_logger_file_handler(self,self.SAVED_DIR / 'log.txt')
        return self