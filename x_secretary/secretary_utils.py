import os
import torch
import torch.distributed as distributed
import random
import numpy as np
from .deprecated import deprecated 
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

@deprecated('This function will be remove in future version, using init_cuda() or init_env() instead.')
def init_cuda_ddp(cuda_devices):
    '''
    setting cuda devices (for ddp)
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_devices
    distributed.init_process_group(backend='nccl')
    LOCAL_RANK=distributed.get_rank()
    WORLD_SIZE=distributed.get_world_size()
    torch.cuda.set_device(LOCAL_RANK)
    return LOCAL_RANK,WORLD_SIZE

def init_cuda(cuda_devices,ddp=False):
    '''
    setting cuda devices

    @return : local rank, world size
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_devices
    if(ddp):
        distributed.init_process_group(backend='nccl')
        LOCAL_RANK=distributed.get_rank()
        WORLD_SIZE=distributed.get_world_size()
        torch.cuda.set_device(LOCAL_RANK)
        return LOCAL_RANK,WORLD_SIZE
    else:
        return 0,1

def set_seed(seed,determinstic=False):
    '''
    set random seed of torch, numpy and random
    '''
    torch.manual_seed(seed)
    random.seed(seed)  # random
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch+CPU
    torch.cuda.manual_seed(seed)  # torch+GPU
    if(determinstic):
        torch.use_deterministic_algorithms(True)

