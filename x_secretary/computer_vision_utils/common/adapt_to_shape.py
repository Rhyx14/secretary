import numpy,einops,cv2
MEAN=[0.485,0.456,0.406]
def adapt_to_shape(img:numpy.ndarray,target_shape:tuple):
    '''
    adapt image to a designated size, keeping the ratio, fill the blank with ImageNet MEAN pixel 
    '''
    assert isinstance(img,numpy.ndarray)
    _oh,_ow,_oc=img.shape
    _dh=abs(_oh-target_shape[0])
    _dw=abs(_ow-target_shape[1])

    _rslt_img=(numpy.array(MEAN[::-1])*255).astype(img.dtype) # mean of ImageNet-1k, bgr
    _rslt_img=einops.repeat(_rslt_img,'c -> h w c', h=target_shape[0],w=target_shape[1])

    if _dh > _dw : # resize the image in the w dimension
        _new_w= target_shape[1]
        _new_h = int(_oh / _ow * _new_w)
        _new_h = min(target_shape[0],_new_h)

    else: # resize the image in the h dimension
        _new_h = target_shape[0]
        _new_w = int(_ow / _oh * _new_h)
        _new_w = min(target_shape[1],_new_w)

    _source_img=cv2.resize(img,(_new_w,_new_h))

    _start_x=int((target_shape[1]-_source_img.shape[1])/2)
    _start_y=int((target_shape[0]-_source_img.shape[0])/2)

    _rslt_img[
        _start_y:_start_y+_source_img.shape[0],
        _start_x:_start_x+_source_img.shape[1],:]=_source_img

    return _rslt_img
