import torch,numpy
from torchvision import transforms
import cv2 
import numpy as np
def flip_segmentation(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        img=torch.flip(img,[-1])
        label=torch.flip(label,[-1])
    return img,label

def extract_segmentation(img,label,stride=2):
    '''
    select pixels with certain stride
    '''
    if img is np.ndarray:
        _img=img[::stride,::stride,:] # [h w c]
    else:
        _img=img[:,::stride,::stride] # [c h w]
    _label=label[::stride,::stride] # [h w]
    return _img,_label


def rot90_segmentation(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        k=torch.randint(1,3,(1,)).item()
        img=torch.rot90(img,k,dims=[1,2])
        label=torch.rot90(label,k)
    return img,label

def opencv_seg_label_to_torchTensor(label: numpy.ndarray) -> torch.Tensor:
    # convert to tensor
    label = torch.from_numpy(label.copy()).long()
    # [h,w,c] -> [c,h,w] -> [h,w]
    label=label.permute(2,0,1)[0]
    return label