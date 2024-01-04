import os
import shutil
import logging
import torch.distributed as dist
import torch
from ..secretary_base import Secretary_base
from pathlib import Path

class Eval_Secretary(Secretary_base):
    def __init__(self,name='Eval_Secretary',saved_dir='.') -> None:
        super().__init__(working_dir=Path(saved_dir),logger_name=name)
        pass
