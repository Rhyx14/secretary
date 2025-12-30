import numpy as np
import cv2
def resize_image_by_short_side(img: np.ndarray, size: int):
    '''
    Resize image and label by short side, keeping the aspect ratio, no padding
    
    Args:
        img: input image as numpy array
        size: target length of the short side
    
    Returns:
        resized_img
    '''
    assert isinstance(img, np.ndarray)

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

    return resized_img