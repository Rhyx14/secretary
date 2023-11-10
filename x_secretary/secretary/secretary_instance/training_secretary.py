import logging,os,uuid
from ..secretary_base import Secretary_base
from ..solo_method import solo_method,solo_chaining_method,solo_method_with_default_return
import torch.distributed as dist
from ...utils.sys_info import get_sys_info
from ...utils.log_dir import Log_dir
import torch
from pathlib import Path
from ...utils.faster_save_on_cpu import offload_module,restore_offload
class Training_Secretary(Secretary_base):
    def __init__(self,distributed=False,saved_dir='.',name_prefix='Train_Secretary',logging_level=logging.INFO) -> None:
        super().__init__(Path(saved_dir),distributed)

        # create folder
        self.Log_dir=Log_dir(
            str(uuid.uuid1()),
            root_path=self.SAVED_DIR,
            distributed=self.distributed
        ).create_dir()
        self.SAVED_DIR=self.Log_dir.saved_dir

        # 保存日志
        self._logging_level=logging_level
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.logger=logging.getLogger(name_prefix)
        self.logger.setLevel(logging_level)
        self._add_logger_file_handler(self.SAVED_DIR/'log.txt')

        self._print_env()

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
        self._add_logger_file_handler(self.SAVED_DIR / 'log.txt')
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

            path=os.path.join(self.SAVED_DIR,file_name)

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

        self.data_recorder.save(self.SAVED_DIR)
        return self

    def offload_module(self,flag, module_type, net, ratio=0):
        '''
        Offload the interim tensor to cpu for reducing VRAM cost 
        '''
        if flag:
            self.warning('Offloading is enabled, may impact the training efficiency.')
            offload_module(module_type,net,ratio)