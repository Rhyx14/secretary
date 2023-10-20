from pathlib import Path
import pathlib
import shutil
import datetime
class Log_dir:
    '''
    创建训练用临时文件夹
    '''
    def __init__(self,name:str,root_path='.',distributed=False) -> None:
        '''
        name: 文件名称

        root_path: where the tmp folder is

        distributed: whether running in torch distibuted parallel mode
        '''
        self.distributed=distributed
        self.name=Path(name)
        self.root_path=Path(root_path)
        if(self.distributed):
            import torch.distributed as dist
            self.local_rank=dist.get_rank()
            self.world_size=dist.get_world_size()
        pass

    def create_dir(self):
        if(self.distributed):
            import torch.distributed as dist
            _saved_dir=self.root_path/self.name

            if(self.local_rank==0):
                pathlib.Path.mkdir(_saved_dir,exist_ok=True)

            ls=[str(_saved_dir)]
            dist.broadcast_object_list(ls,0)
            self.saved_dir=Path(ls[0]) 

        else:
            self.saved_dir=self.root_path/self.name
            pathlib.Path.mkdir(self.saved_dir,exist_ok=True)
        return self
    
    def change_name(self,name):
        '''
        change the name of the directory
        '''
        if self.distributed:
            import torch.distributed as dist
            _new_name='./asdf'
            if (self.local_rank==0):
                _new_name=self.root_path / name
                shutil.move(self.saved_dir,_new_name)

            ls=[str(_new_name)]
            dist.broadcast_object_list(ls,0)
            _new_name=Path(ls[0])
        else:
            _new_name=self.root_path / name
            shutil.move(self.saved_dir,_new_name)
    
        self.name=name
        self.saved_dir=_new_name


    @staticmethod
    def time_suffix_name(name)->str:
        '''
        Generate a name with timestamp suffix
         
        name: prefix
        '''
        return name+datetime.datetime.now().strftime('_%m%d%H%M')