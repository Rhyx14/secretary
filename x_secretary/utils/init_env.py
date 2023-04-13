import os
import torch
import torch.distributed as distributed
from ..deprecated import deprecated 
from .set_seeds import set_seed

@deprecated('This function will be remove in future version, using init_cuda() instead.')
def init_env(cuda_devices,ddp=False,openmpi_thread=None,random_seed=None):
    '''
    initilize enviroment (for cuda)
    '''
    if(openmpi_thread is not None):
        os.environ['OMP_NUM_THREADS']=openmpi_thread
    if random_seed is not None:
        set_seed(random_seed)
    local_rank,world_size=init_cuda(cuda_devices,ddp)
    return local_rank,world_size

def init_cuda(cuda_devices,ddp=False,tf32=False):
    '''
    setting cuda devices and environment.

    @return : local rank, world size
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_devices

    if tf32 is not True:
        os.environ['NVIDIA_TF32_OVERRIDE']='0'

    if(ddp):
        distributed.init_process_group(backend='nccl')
        LOCAL_RANK=distributed.get_rank()
        WORLD_SIZE=distributed.get_world_size()
        torch.cuda.set_device(LOCAL_RANK)
        return LOCAL_RANK,WORLD_SIZE
    else:
        return 0,1


