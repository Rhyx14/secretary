import torch
from torchvision.transforms.v2.functional import resize_image
import torch.distributed as dist
class AlternatingResize(torch.nn.Module):
    def __init__(self,shape_list,frequency) -> None:
        super().__init__()
        self._index=0
        self._rank=dist.get_rank()
        
        self._shape_list=shape_list

        if isinstance(frequency,int):
            self._freqency_list=[frequency for _ in range(frequency)]
        else:
            assert len(frequency)==len(self._shape_list)
            self._freqency_list=frequency

        self._expanded_shape_list=[]

        for _shape,_time in zip(self._shape_list,self._freqency_list):
            if isinstance(_shape,int):
                _shape=(_shape,_shape)
            self._expanded_shape_list.extend([_shape for _ in range(_time)])
        
        self._current_shape=self._expanded_shape_list[self._index]
    
    def next_shape(self):
        self._current_shape=self._expanded_shape_list[self._index]
        self._index=(self._index+1) % len(self._expanded_shape_list)
    
    def forward(self,x):
        return resize_image(x,self._current_shape)
    
    def extra_repr(self) -> str:
        return f'shapes={self._shape_list}, freq={self._freqency_list}'