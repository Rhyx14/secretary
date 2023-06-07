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
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if(ddp):
        distributed.init_process_group(backend='nccl')
        LOCAL_RANK=distributed.get_rank()
        WORLD_SIZE=distributed.get_world_size()
        torch.cuda.set_device(LOCAL_RANK)
        return LOCAL_RANK,WORLD_SIZE
    else:
        return 0,1


