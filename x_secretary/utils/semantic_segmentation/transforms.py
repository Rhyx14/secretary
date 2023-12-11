import torch,numpy
from torchvision import transforms
import cv2 

def crop(img: torch.Tensor | numpy.ndarray,crop_factor=32,down_size=1):
    if isinstance(img,torch.Tensor):
        # downsize
        img=img[...,::down_size,::down_size]

        h, w = img.shape[-2],img.shape[-1]
        _new_h= ((h//crop_factor))*crop_factor
        _new_w= ((w//crop_factor))*crop_factor

        # crop
        img=transforms.CenterCrop((_new_h,_new_w))(img)

    elif isinstance(img,numpy.ndarray):        
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

def opencv_to_torchTensor(img:numpy.ndarray) -> torch.Tensor:
    '''
    From opencv images [h,w,c] to pytorch tensor [c,h,w]
    
    [0-255] -> [0.,1.]
    '''
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # [h,w,c]
    # convert to tensor
    img:torch.Tensor = torch.from_numpy(img.copy()).float()
    # [h,w,c] -> [c,h,w]
    img=img.permute(2,0,1)
    img=img/255.
    return img