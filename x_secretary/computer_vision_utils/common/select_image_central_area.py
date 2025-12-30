import numpy,torch
def select_image_central_area(img,shape:tuple):
    if isinstance(img,numpy.ndarray):
        h,w,c=img.shape
        start_h=(h-shape[0]) //2
        start_w=(w-shape[1]) //2
        return img[start_h:start_h+shape[0],start_w:start_w+shape[1],:]
    elif isinstance(img,torch.Tensor):
        c,h,w=img.shape
        start_h=(h-shape[0]) //2
        start_w=(w-shape[1]) //2
        return img[:,start_h:start_h+shape[0],start_w:start_w+shape[1]]
