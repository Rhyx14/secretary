import numpy,einops,cv2
# RGB
MEAN=[0.485,0.456,0.406]
def adapt_to_shape_unified(img: numpy.ndarray, target_shape: tuple):
    '''
    adapt image to a designated size, keeping the ratio, fill the blank with ImageNet MEAN pixel

    note that the label should be a three channel image (but all channel are the same )

    Unified version that works for both enlarging and shrinking
    '''
    assert isinstance(img, numpy.ndarray)
    _oh, _ow, _oc = img.shape
    
    _rslt_img = (numpy.array(MEAN[::-1]) * 255).astype(img.dtype)
    _rslt_img = einops.repeat(_rslt_img, 'c -> h w c', h=target_shape[0], w=target_shape[1])

    # 统一逻辑：基于缩放比例
    scale_h = target_shape[0] / _oh
    scale_w = target_shape[1] / _ow
    
    # 选择较小的缩放比例，确保图像完全在目标范围内
    scale = min(scale_h, scale_w)
    
    _new_w = int(_ow * scale)
    _new_h = int(_oh * scale)

    _source_img = cv2.resize(img, (_new_w, _new_h))

    _start_x = (target_shape[1] - _new_w) // 2
    _start_y = (target_shape[0] - _new_h) // 2

    _rslt_img[_start_y:_start_y+_new_h, _start_x:_start_x+_new_w, :] = _source_img

    return _rslt_img