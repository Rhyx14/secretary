import random
import numpy as np
import torch


def set_seed(seed,determinstic=True):
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
