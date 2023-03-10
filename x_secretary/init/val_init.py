import datetime
import os
import shutil
from ..secretary_solo_method import solo_method
import logging
import torch.distributed as dist
import torch
from .init_base import init_base

class Val_init(init_base):
    def __init__(self,cfg,weight_folder) -> None:
        self.cfg=cfg
        self.logger=logging.getLogger(cfg.NAME)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.WEIGHT_DIR=weight_folder

        self._updated={
            'cfg':self.cfg,
            'logger':self.logger
        }
        self.logger.setLevel(logging.INFO)
        pass
