from typing import Callable,Iterable
import os,uuid,sys,torch
from ..secretary_base import Secretary_base
from ..solo_method import solo_method,solo_chaining_method,solo_method_with_default_return
from ...utils.sys_info import get_sys_info
from ...utils.log_dir import Log_dir
from loguru import logger
from pathlib import Path
from ...utils.faster_save_on_cpu import offload_module,restore_offload
from ...configuration.configuration import Configuration
LOGGER_FMT='<blue>{time:YYYY-MM-DD HH:mm:ss Z}</blue> [<level>{level}</level>] <green>{name}:{line}</green><yellow>#</yellow> {message}'
class Training_Secretary(Secretary_base):
    def __init__(self,
                 saved_dir='.',
                 logging_level='INFO', 
                 saving_main_script=True,
                 log_environment=True) -> None:
        super().__init__(Path(saved_dir))

        # create folder
        self.Log_dir=Log_dir(
            str(uuid.uuid1()),
            root_path=self._working_dir,
            distributed=self._distributed
        ).create_dir()
        self._working_dir=self.Log_dir.dir()

        logger.remove()
        logger.add(sys.stderr,format=LOGGER_FMT)
        # 保存日志
        self._log_file_handler=logger.add(str(self._working_dir/'log.txt'),level=logging_level,format=LOGGER_FMT)
        self._logging_level=logging_level

        # 保存启动脚本
        if saving_main_script:
            self.save_main_script()

        # 打印环境信息
        if log_environment:
            self._log_env() 
    
    def _saving_file(self,source:Path,name:str=None):
        if name is None:
            (self._working_dir/ source.name).write_text(Path.read_text(source),encoding='utf-8')
        else:
            (self._working_dir / name).write_text(Path.read_text(source),encoding='utf-8')

    @solo_chaining_method
    def saving_files(self,path: Path | Iterable[Path] | tuple | Iterable[tuple[str | str]]):
        '''
        Copy files (text) to working dir

        path: Pathlib.Path, if using the original name;

              (Pathlib.Path,Str), rename to Str
        '''
        if isinstance(path,Path):
            self._saving_file(path)
        elif isinstance(path,tuple):
            self._saving_file(*path)
        else:
            for _p in path:
                if isinstance(_p, Path):
                    self._saving_file(_p)
                else:
                    self._saving_file(*_p)
        return self

    @solo_chaining_method
    def save_main_script(self):
        '''
        Saving the running script
        '''
        self._saving_file(Path(sys.argv[0]))
        return self

    @solo_method
    def _log_env(self):
        '''
        ouput the environment

        saving to 'env.txt' 
        '''
        # print(cfg_str)
        with open(self._working_dir/'env.txt','w') as f:
            f.write(get_sys_info())

    def _log_to_file(self,content,file_name,prefix):
        with open(self._working_dir/file_name,'a+') as f:
            if not isinstance(prefix,str): prefix=prefix()
            if not isinstance(content,str): content=content()
            if prefix !='': f.write(f'# {prefix}\n')
            f.write(content)
            f.write('\n')
        return

    @solo_chaining_method
    def log(self,content: str | Callable[..., str] ,file_name: str,prefix: str | Callable[..., str] =''):
        '''
        log text contents to designated file in the working directory
        '''
        self._log_to_file(content,file_name,prefix)
        return self

    @solo_chaining_method
    def log_to_cfg(self,content: str | Callable[..., str],prefix : str | Callable[..., str] =''):
        '''
        log string to the configuration files,
        '''
        self._log_to_file(content,'configuration.md', prefix)
        return self
    
    @solo_chaining_method
    def log_cfg_changes(self,cfg:Configuration,prefix: str | Callable[..., str] ='',reset=True):
        '''
        log string to the changes of configuration objects,

        note that this function is only suitable for the Stage Mode
        '''
        self._log_to_file(cfg.get_records_str(),'configuration.md', prefix)
        if reset: cfg.reset_records()
        return self
        
    def set_working_dir_name(self,name_prefix: str):
        '''
        change the working dir's name
        '''
        self.sync()
        self.Log_dir.change_name(Log_dir.time_suffix_name(name_prefix))
        self._working_dir=self.Log_dir.saved_dir
        logger.remove(self._log_file_handler)
        self._log_file_handler=logger.add(str(self._working_dir/'log.txt'),level=self._logging_level,format=LOGGER_FMT)
        self.sync()
        return self
    
    @solo_chaining_method
    def save(self,net: torch.nn.Module=None,best_mode=False,best_value=None,file_name='weight.pt'):
        '''
        save network weight and recorded data (solo) -> self

        if net is None, save the data only

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
                    logger.info(f'invaild best_value, pass')
                else:
                    if hasattr(self,'_best_value'):
                        if best_value>self._best_value:
                            self._best_value=best_value
                            torch.save(_net.state_dict(),path)
                            logger.info(f'saved at {path}')
                        else:
                            logger.info(f'not the best ({self._best_value}), pass')
                    else: # the first time saving the best
                        setattr(self,'_best_value',best_value)
                        torch.save(_net.state_dict(),path)
                        logger.info(f'saved at {path}')
            else: 
                torch.save(_net.state_dict(),path)
                logger.info(f'saved at {path}')

        self._data.save(self._working_dir)
        return self

    def offload_module(self,flag: bool, module_type, net: torch.nn.Module, ratio=1. ):
        '''
        Offload the interim tensor to cpu for reducing VRAM cost 

        ratio: the precentage of offloaded tensors
        '''
        if flag:
            logger.warning('Offloading is enabled, may impact the training efficiency.')
            offload_module(module_type,net,ratio)


