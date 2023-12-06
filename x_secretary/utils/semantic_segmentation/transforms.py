import torch
from torchvision import transforms
def crop(img,crop_factor=32,down_size=1):
    h, w = img.shape[-2],img.shape[-1]
    
    _new_h= ((h//down_size)//crop_factor)*crop_factor
    _new_w= ((w//down_size)//crop_factor)*crop_factor

    # downsize
    img=img[...,::down_size,::down_size]
    # crop
    img=transforms.CenterCrop((_new_h,_new_w))(img)
    return img

def flip(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        img=torch.flip(img,[-1])
        label=torch.flip(label,[-1])
    return img,label

def rot90(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        k=torch.randint(1,3,(1,)).item()
        img=torch.rot90(img,k,dims=[1,2])
        label=torch.rot90(label,k)
    return img,label