import os
import shutil
import logging
import torch.distributed as dist
import torch
from ..secretary_base import Secretary_base
from pathlib import Path

class Eval_Secretary(Secretary_base):
    def __init__(self,name='Eval_Secretary',saved_dir='.',distributed=False) -> None:
        super().__init__(saved_dir=Path(saved_dir),distributed=distributed)

        self.logger=logging.getLogger(name)
        logging.basicConfig(level=logging.INFO,format='%(asctime)s-[%(name)s] %(message)s')
        self.logger.setLevel(logging.INFO)
        pass
