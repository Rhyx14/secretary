import cv2
from PIL import Image
class OpenCV_Loader:
    RET_PIL=0
    RET_OpenCV=1
    def __init__(self,ret_type,resize:tuple=None) -> None:
        self.resize=resize
        self.ret_type=ret_type

    def __call__(self, path):
        img=cv2.imread(path,cv2.IMREAD_COLOR)
        if self.resize[0] is not None:
            img=cv2.resize(img,dsize=self.resize)
        if self.ret_type==OpenCV_Loader.RET_PIL:
            return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        elif self.ret_type==OpenCV_Loader.RET_OpenCV:
            return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        else:
            raise ValueError