import random
import numpy as np
import torch

import random
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    '''
    set workers' seed for dataloader
    '''
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    np.random.seed(worker_id)
    random.seed(worker_id)

def get_generator(seed=0):
    '''
    set dataloader's generator
    '''
    g = torch.Generator()
    g.manual_seed(seed)
