import torch
from loguru import logger
from torchvision.transforms.v2.functional import resize_image
import torch.distributed as dist
class AlternatingResize(torch.nn.Module):
    '''
    Resize the image with certain shapes

    Need to maunally set the next_shape() to alter the target shape 

    Only support torch.Tensor, i.e., transforms.v2, in the extra_transform part
    '''
    def __init__(self,shape_list,frequency) -> None:
        super().__init__()
        self._index=0

        self._shape_list=shape_list

        if isinstance(frequency,int):
            self._freqency_list=[frequency for _ in range(len(shape_list))]
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
        logger.info(f'resize to {self._current_shape}')

    def forward(self,*args):
        if len(args)==1:
            return resize_image(args[0],self._current_shape)
        else:
            return resize_image(args[0],self._current_shape),*args[1:]
    
    def extra_repr(self) -> str:
        return f'shapes={self._shape_list}, freq={self._freqency_list}'