import random,cv2,torch
import numpy as np
from torchvision import transforms

def crop(img: torch.Tensor | np.ndarray,crop_factor=32,down_size=1):
    if isinstance(img,torch.Tensor):
        # downsize
        img=img[...,::down_size,::down_size]

        h, w = img.shape[-2],img.shape[-1]
        _new_h= ((h//crop_factor))*crop_factor
        _new_w= ((w//crop_factor))*crop_factor

        # crop
        img=transforms.CenterCrop((_new_h,_new_w))(img)

    elif isinstance(img,np.ndarray):        
        # downsize
        img=img[::down_size,::down_size,...]

        h,w = img.shape[0],img.shape[1]
        _new_h= ((h//crop_factor))*crop_factor
        _new_w= ((w//crop_factor))*crop_factor

        # crop
        _start_h=(h-_new_h)//2
        _start_w=(w-_new_w)//2
        img=img[_start_h:_start_h+_new_h, 
                _start_w:_start_w+_new_w]

    return img