from typing import Any
from .detection_dataset import Dectection_Dataset
import torch
from tqdm import tqdm
from itertools import product
from .nms import nms
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

    def _decoder(self,pred):
        '''
        pred (tensor) [1,G,G,30]
        return (tensor) box[[x1,y1,x2,y2]] label[...]
        '''

        grid_num = 14
        
        boxes=[]
        category=[]
        probs = []

        cell_size = 1./grid_num

        pred=pred.squeeze(0) # -> [G,G,30]
        contain1 = pred[:,:,4].unsqueeze(2) # confidence of box1 [G,G,1]
        contain2 = pred[:,:,9].unsqueeze(2) # confidence of box2 [G,G,1]
        contain = torch.cat((contain1,contain2),2) # [G,G,2]
        mask1 = contain > 0.1 #大于阈值
        mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
        mask = (mask1+mask2).gt(0) # still [G,G,2], equivalent to 1 means selected bbox

        for i,j,b in product(range(grid_num),range(grid_num),range(2)):  # for 2 bbox in the output
            if mask[i,j,b] == 1: # if the selected
                box = pred[i,j,b*5:b*5+4] # [4] 
                contain_prob = torch.FloatTensor([pred[i,j,b*5+4]]) # confidence

                xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                box[:2] = box[:2]*cell_size + xy # 获取box 的x,y关于整个图像的坐标，return cxcy relative to image, 即框中心点的坐标

                box_xy = torch.FloatTensor(4) #转换成xy形式（关于图像的左上右下bbox表示） convert[cx,cy,w,h] to [x1,y1,x2,y2]
                box_xy[:2] = box[:2] - 0.5*box[2:]
                box_xy[2:] = box[:2] + 0.5*box[2:]

                # TODO 可优化，不用tensor运算
                max_prob,cls_index = torch.max(pred[i,j,10:],0)
                if float((contain_prob*max_prob)[0]) > 0.1:
                    boxes.append(box_xy.view(1,4))
                    category.append(cls_index)
                    probs.append(contain_prob*max_prob)

        if len(boxes) ==0:
            boxes = torch.zeros((1,4))
            probs = torch.zeros(1)
            category = torch.zeros(1)
        else:
            boxes = torch.cat(boxes,0) #(n,4)
            probs = torch.cat(probs,0) #(n,)
            category= torch.Tensor(category)
            # cls_indexs = torch.cat(cls_indexs,0) #(n,)
        keep = nms(boxes,probs)
        return boxes[keep],category[keep],probs[keep]