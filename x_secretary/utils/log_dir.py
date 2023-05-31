from pathlib import Path
import pathlib
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

    def create_dir(self) -> Log_dir:
        if(self.distributed):
            import torch.distributed as dist
            _saved_dir=str(self.root_path/self.name)

            if(self.local_rank==0):
                pathlib.Path.mkdir(_saved_dir,exist_ok=True)

            ls=[_saved_dir]
            dist.broadcast_object_list(ls,0)
            self.saved_dir=Path(ls[0]) 

        else:
            self.saved_dir=self.root_path/self.name
            pathlib.Path.mkdir(self.saved_dir,exist_ok=True)
        return self

    @staticmethod
    def time_suffix_name(name)->str:
        '''
        Generate a name with timestamp suffix
         
        name: prefix
        '''
        name+datetime.datetime.now().strftime('_%m%d%H%M')
        return 