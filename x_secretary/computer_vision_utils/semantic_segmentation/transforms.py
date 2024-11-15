import torch,numpy
from torchvision import transforms
import cv2 
import numpy as np
import einops
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

# RGB
MEAN=[0.485,0.456,0.406]
def adapt_to_shape(img:np.ndarray,label:np.ndarray,target_shape:tuple,ignore_label=255):
    '''
    adapt image to a designated size, keeping the ratio, fill the blank with ImageNet MEAN pixel 

    note that the label should be a three channel image (but all channel are the same )
    '''
    assert isinstance(img,np.ndarray)
    assert isinstance(label,np.ndarray)
    _oh,_ow,_oc=img.shape
    _dh=abs(_oh-target_shape[0])
    _dw=abs(_ow-target_shape[1])

    _rslt_img=(np.array(MEAN[::-1])*255).astype(img.dtype) # mean of ImageNet-1k, bgr
    _rslt_img=einops.repeat(_rslt_img,'c -> h w c', h=target_shape[0],w=target_shape[1])
    _rslt_label=np.zeros_like(_rslt_img,dtype=label.dtype)+255

    if _dh > _dw : # resize the image in the w dimension
        _new_w= target_shape[1]
        _new_h = int(_oh / _ow * _new_w)
        _new_h = min(target_shape[0],_new_h)

    else: # resize the image in the h dimension
        _new_h = target_shape[0]
        _new_w = int(_ow / _oh * _new_h)
        _new_w = min(target_shape[1],_new_w)

    _source_img=cv2.resize(img,(_new_w,_new_h))
    _source_label=cv2.resize(label,(_new_w,_new_h),interpolation=cv2.INTER_NEAREST)

    _start_x=int((target_shape[1]-_source_img.shape[1])/2)
    _start_y=int((target_shape[0]-_source_img.shape[0])/2)

    _rslt_img[
        _start_y:_start_y+_source_img.shape[0],
        _start_x:_start_x+_source_img.shape[1],:]=_source_img
    _rslt_label[
        _start_y:_start_y+_source_label.shape[0],
        _start_x:_start_x+_source_label.shape[1],...]=_source_label
    
    return _rslt_img,_rslt_label

import random
def select_random_area(img,label,shape=(128,128)):
    '''
    randomly select area
    '''
    if isinstance(img,np.ndarray):
        h,w,c=img.shape
        start_h=random.randrange(0,h-shape[0])
        start_w=random.randrange(0,w-shape[1])
        return img[start_h:start_h+shape[0],start_w:start_w+shape[1],:],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]
    elif isinstance(img,torch.Tensor):
        c,h,w=img.shape
        start_h=random.randrange(0,h-shape[0])
        start_w=random.randrange(0,w-shape[1])
        return img[:,start_h:start_h+shape[0],start_w:start_w+shape[1]],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]

def select_central_area(img,label,shape=(128,128)):
    '''
    select central area 
    '''
    if isinstance(img,np.ndarray):
        h,w,c=img.shape
        start_h=(h-shape[0]) //2
        start_w=(h-shape[1]) //2
        return img[start_h:start_h+shape[0],start_w:start_w+shape[1],:],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]
    elif isinstance(img,torch.Tensor):
        c,h,w=img.shape
        start_h=(h-shape[0]) //2
        start_w=(h-shape[1]) //2
        return img[:,start_h:start_h+shape[0],start_w:start_w+shape[1]],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]