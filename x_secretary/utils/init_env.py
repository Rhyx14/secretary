import os
import torch
import torch.distributed as distributed
from ..deprecated import deprecated 

def init_cuda(cuda_devices,ddp=False,tf32=False):
    '''
    setting cuda devices and environment.

    @return : local rank, world size
    '''
    os.environ['CUDA_VISIBLE_DEVICES']=cuda_devices
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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


