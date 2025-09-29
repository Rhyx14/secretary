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
    '''
    From opencv images [h,w,c] (BGR) to pytorch tensor [c,h,w] (RGB)
    
    [0-255] -> [0.,1.]

    The label image should have the same value among all channels. Here the method will only return the first channel,

    i.e., [c h w] -> [0 h w]
    '''
    # convert to tensor
    label = torch.from_numpy(label.copy()).long()
    # [h,w,c] -> [c,h,w] -> [h,w]
    label=label.permute(2,0,1)[0]
    return label

def resize_by_short_side(img: np.ndarray, label: np.ndarray, size: int):
    '''
    Resize image and label by short side, keeping the aspect ratio, no padding
    
    Args:
        img: input image as numpy array
        label: label as numpy array (3 channels but all channels are the same)
        size: target length of the short side
    
    Returns:
        resized_img, resized_label
    '''
    assert isinstance(img, np.ndarray)
    assert isinstance(label, np.ndarray)
    
    h, w = img.shape[:2]
    
    # Determine the scaling factor based on the short side
    if h < w:  # height is the short side
        new_h = size
        new_w = int(w * size / h)
    else:  # width is the short side or equal
        new_w = size
        new_h = int(h * size / w)
    
    # Resize image and label
    resized_img = cv2.resize(img, (new_w, new_h))
    resized_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    return resized_img, resized_label

# RGB
MEAN=[0.485,0.456,0.406]
def adapt_to_shape_unified(img: np.ndarray, label: np.ndarray, target_shape: tuple, ignore_label=255):
    '''
    adapt image to a designated size, keeping the ratio, fill the blank with ImageNet MEAN pixel

    note that the label should be a three channel image (but all channel are the same )

    Unified version that works for both enlarging and shrinking
    '''
    assert isinstance(img, np.ndarray)
    assert isinstance(label, np.ndarray)
    _oh, _ow, _oc = img.shape
    
    _rslt_img = (np.array(MEAN[::-1]) * 255).astype(img.dtype)
    _rslt_img = einops.repeat(_rslt_img, 'c -> h w c', h=target_shape[0], w=target_shape[1])
    _rslt_label = np.zeros_like(_rslt_img, dtype=label.dtype) + ignore_label

    # 统一逻辑：基于缩放比例
    scale_h = target_shape[0] / _oh
    scale_w = target_shape[1] / _ow
    
    # 选择较小的缩放比例，确保图像完全在目标范围内
    scale = min(scale_h, scale_w)
    
    _new_w = int(_ow * scale)
    _new_h = int(_oh * scale)

    _source_img = cv2.resize(img, (_new_w, _new_h))
    _source_label = cv2.resize(label, (_new_w, _new_h), interpolation=cv2.INTER_NEAREST)

    _start_x = (target_shape[1] - _new_w) // 2
    _start_y = (target_shape[0] - _new_h) // 2

    _rslt_img[_start_y:_start_y+_new_h, _start_x:_start_x+_new_w, :] = _source_img
    _rslt_label[_start_y:_start_y+_new_h, _start_x:_start_x+_new_w, :] = _source_label
    
    return _rslt_img, _rslt_label

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
        start_w=(w-shape[1]) //2
        return img[start_h:start_h+shape[0],start_w:start_w+shape[1],:],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]
    elif isinstance(img,torch.Tensor):
        c,h,w=img.shape
        start_h=(h-shape[0]) //2
        start_w=(w-shape[1]) //2
        return img[:,start_h:start_h+shape[0],start_w:start_w+shape[1]],label[start_h:start_h+shape[0],start_w:start_w+shape[1]]