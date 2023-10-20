import datetime
import os
import shutil
import logging
import torch.distributed as dist
import torch
from .init_base import init_base

class Val_init(init_base):
    def __init__(self,cfg) -> None:
        self.cfg=cfg
        self.logger=logging.getLogger(cfg.NAME)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')

        self._updated={
            'cfg':self.cfg,
            'logger':self.logger,
            'LOCAL_RANK':0,
        }

        if torch.distributed.is_torchelastic_launched():
            self._updated |={
                'LOCAL_RANK':torch.distributed.get_rank(),
                'distributed':True
            }

        self.logger.setLevel(logging.INFO)
        pass
