import random
import cv2
import numpy as np
import torch
from torchvision import transforms
from .random_function import random_function

@random_function
def blur(img,p=0.5):
    img = cv2.blur(img,(5,5))
    return img

@random_function
def RandomHSV(img:np.ndarray,p=0.5,mode='saturation'):
    adjust = random.choice([0.5,1.5])
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    match mode:
        case 'hue'       : hsv[...,0] = hsv*adjust
        case 'saturation': hsv[...,1] = hsv*adjust
        case 'brightness': hsv[...,2] = hsv*adjust

    hsv = np.clip(hsv, 0, 255).astype(img.dtype)
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return img

def random_saturation(img:np.ndarray,p=0.5):
    return RandomHSV(img,p=p,mode='saturation')

def random_brightness(img:np.ndarray,p=0.5):
    return RandomHSV(img,p=p,mode='brightness')

def random_hue(img:np.ndarray,p=0.5):
    return RandomHSV(img,p=p,mode='hue')

def opencv_to_torchTensor(img:np.ndarray) -> torch.Tensor:
    '''
    From opencv images [h,w,c] (BGR) to pytorch tensor [c,h,w] (RGB)
    
    [0-255] -> [0.,1.]
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # [h,w,c]
    # convert to tensor
    img:torch.Tensor = torch.from_numpy(img.copy()).float()
    # [h,w,c] -> [c,h,w]
    img=img.permute(2,0,1)
    img=img/255.
    return img

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