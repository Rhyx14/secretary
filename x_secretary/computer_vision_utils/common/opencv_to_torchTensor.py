import cv2
import numpy as np
import torch
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