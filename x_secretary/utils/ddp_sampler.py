import torch
from typing import Iterable, Iterator, Optional, Sized,List,Union
from torch.utils.data.sampler import Sampler,RandomSampler,SequentialSampler
import torch.distributed as distributed
class DDP_BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self,dataset,shuffle, batch_size: int,seed=0) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        if(shuffle):
            generator=torch.Generator()
            generator.manual_seed(seed)
            self.sampler = RandomSampler(dataset,generator=generator)
        else:
            raise NotImplementedError
        self.batch_size = batch_size
       
        self.local_rank=distributed.get_rank()
        self.world_size=distributed.get_world_size()
        self._span=batch_size//self.world_size

    def __iter__(self) -> Iterator[List[int]]:

        _begin=self.local_rank*self._span
        _end=_begin+self._span

        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch[_begin:_end]
                batch = []

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
