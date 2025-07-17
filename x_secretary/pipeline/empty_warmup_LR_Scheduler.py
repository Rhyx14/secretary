# from pytorch_warmup.base import BaseWarmup,_check_optimizer,get_warmup_params
from contextlib import contextmanager
class EmptyLRScheduler():
    def __init__(self) -> None:
        pass
    def step(self):
        pass
    
class EmptyWarmup():
    """empty warmup schedule.

    The empty warmup schedule (a no-warmup placeholder)

    do nothing here
    """
    @contextmanager
    def dampening(self):
        yield
