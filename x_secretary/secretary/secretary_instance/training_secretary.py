import logging,os,uuid
from ..secretary_base import Secretary_base
from ..solo_method import solo_method,solo_chaining_method,solo_method_with_default_return
import torch.distributed as dist
from ...utils.sys_info import get_sys_info
from ...utils.log_dir import Log_dir
import torch
from pathlib import Path
from ...utils.faster_save_on_cpu import offload_module,restore_offload
from ...configuration import Configuration
class Training_Secretary(Secretary_base):
    def __init__(self,saved_dir='.',name_prefix='Training_Secretary',logging_level=logging.INFO) -> None:
        super().__init__(Path(saved_dir),logger_name=name_prefix,logging_level=logging_level)

        # create folder
        self.Log_dir=Log_dir(
            str(uuid.uuid1()),
            root_path=self._working_dir,
            distributed=self._distributed
        ).create_dir()

        # 保存日志
        self._add_logger_file_handler(self.Log_dir.dir()/'log.txt',logging_level)

        # 打印环境信息
        self._log_env()

    @solo_method
    def _log_env(self):
        '''
        ouput the environment

        saving to 'env.txt' 
        '''
        # print(cfg_str)
        with open(self.Log_dir.dir()/'env.txt','w') as f:
            f.write(get_sys_info())

    @solo_chaining_method
    def log_to_cfg(self,s,prefix=''):
        '''
        log string to the configuration files,

        s should be str or callable object 
        '''
        with open(self._working_dir/'configuration.txt','a') as f:
            if prefix !='':
                f.write(prefix)
                f.write('\n')
            if not isinstance(s,str):
                s=s()
            f.write(s)
            f.write('\n')
        return self
    
    @solo_chaining_method
    def log_cfg_changes(self,cfg:Configuration,name='',reset=True):
        '''
        log string to the changes of configuration objects,

        note that this function is only suitable for the Stage Mode
        '''
        with open(self._working_dir/'configuration.txt','a') as f:
            f.write(name+'\n')
            f.write(cfg.get_records_str())
            if reset:
                cfg.reset_records()
            f.write('\n')
        return self
        
    def _add_logger_file_handler(self,path,level):
        if hasattr(self,'_logger_filehandler'):
            self.logger.removeHandler(self._logger_filehandler)
        self._logger_filehandler=logging.FileHandler(path)
        self._logger_filehandler.setFormatter(self._default_logging_formatter)
        self._logger_filehandler.setLevel(self._logging_level)
        self._logger.addHandler(self._logger_filehandler)
        pass
        
    def set_name_prefix(self,name_prefix):
        self.Log_dir.change_name(Log_dir.time_suffix_name(name_prefix))
        self._logger.name=name_prefix
        self._working_dir=self.Log_dir.saved_dir
        self._add_logger_file_handler(self._working_dir / 'log.txt',self._logging_level)
        return self
    
    @solo_chaining_method
    def save(self,net=None,best_mode=False,best_value=None,file_name='weight.pt'):
        '''
        save network weight and recorded data (solo) -> self

        If best_mode == False, the literal value of the best_value doesn't make a difference.
        otherwise, the best_mode is only enabled with a valid best_value.
        '''
        if net is not None:
            if isinstance(net,(torch.nn.parallel.DataParallel,torch.nn.parallel.distributed.DistributedDataParallel)):
                _net=net.module
            else:
                _net=net

            path=os.path.join(self._working_dir,file_name)

            if best_mode:

                if best_value is None:
                    self.logger.info(f'invaild best_value, pass')
                else:
                    if hasattr(self,'_best_value'):
                        if best_value>self._best_value:
                            self._best_value=best_value
                            torch.save(_net.state_dict(),path)
                            self.logger.info(f'saved at {path}')
                        else:
                            self.logger.info(f'not the best ({self._best_value}), pass')
                    else: # the first time saving the best
                        setattr(self,'_best_value',best_value)
                        torch.save(_net.state_dict(),path)
                        self.logger.info(f'saved at {path}')
            else: 
                torch.save(_net.state_dict(),path)
                self.logger.info(f'saved at {path}')

        self._data.save(self._working_dir)
        return self

    def offload_module(self,flag, module_type, net, ratio=0):
        '''
        Offload the interim tensor to cpu for reducing VRAM cost 
        '''
        if flag:
            self.warning('Offloading is enabled, may impact the training efficiency.')
            offload_module(module_type,net,ratio)


