from typing import Any
from .detection_dataset import Dectection_Dataset
import torch
from tqdm import tqdm
from .metric import eval
from collections import defaultdict
class Img_Info_Hooks():
    def __init__(self) -> None:
        pass

    def __call__(self, img) -> Any:
        self.original_size=img.shape
        return img

class Detection_Evaluator():
    def __init__(self,model,COLOR_TABEL) -> None:
        self._model=model
        self._COLOR_TABLE=COLOR_TABEL
        pass

    def eval_dataset(self,dataset:Dectection_Dataset,default_device:str,on_prediction_begin=None):

        self._img_info_hooks=Img_Info_Hooks()
        dataset.union_transform.transforms.insert(0,[self._img_info_hooks,None,None]) # 需要添加hooks，保存原始图像的长宽高

        with torch.no_grad():
            targets =  defaultdict(list)
            preds = defaultdict(list)
            for id,(img,boxes,labels) in enumerate(tqdm(dataset,leave=False)):

                if on_prediction_begin is not None:
                    on_prediction_begin(id,img,boxes,labels)

                _pred=self.predict_gpu(img.unsqueeze(0),default_device)

                for (x1,y1),(x2,y2),category,prob in _pred:
                    preds[self._COLOR_TABLE[category]].append([id,prob,x1,y1,x2,y2])
                for i,_l in enumerate(labels):
                    targets[(id,self._COLOR_TABLE[_l])].append(list(boxes[i]))
                pass
            
            eval(preds,targets,self._COLOR_TABLE)


    def predict_gpu(self,img:torch.Tensor,default_device):
        h,w,c = self._img_info_hooks.original_size 
        with torch.no_grad():
            img = img.to(default_device)
            pred = self._model(img) #[1,g,g,30]
            pred = pred.cpu()

        boxes,cls_indexs,probs =  self._decoder(pred)

        result=[]
        for i,box in enumerate(boxes):
            x1,x2 = int(box[0]*w),int(box[2]*w)
            y1,y2 = int(box[1]*h),int(box[3]*h)
            # x1,x2,y1,y2 = box[0],box[2],box[1],box[3]
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1,y1),(x2,y2),cls_index,prob])

        return result

    def _decoder(self,pred) -> list[ list[tuple,tuple,int,float]]:
        """_summary_

        Args:
            pred (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            list[ list[tuple,tuple,int,float]]: return lefttop coordinate, right bottom coordinate
        """
        raise NotImplementedError