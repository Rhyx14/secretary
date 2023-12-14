import os,sys,random,fileinput,torch,cv2
from pathlib import Path
import torch.utils.data as data
from x_secretary.computer_vision_utils.union_transforms import Union_Transforms
class Dectection_Dataset(data.Dataset):
    def __init__(self,root_dir,file_list,transforms):

        self.root_dir=Path(root_dir)

        self._fnames = []
        self._boxes = []
        self._labels = []

        self.union_transform=Union_Transforms(transforms)

        with fileinput.input(map(lambda f: str(self.root_dir / f),file_list)) as file:
            for line in file:
                if(line == '' ): continue
                # txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
                splited = line.strip().split()
                self._fnames.append(splited[0])
                num_boxes = (len(splited) - 1) // 5
                box=[]
                label=[]
                for i in range(num_boxes):
                    x = float(splited[1+5*i])
                    y = float(splited[2+5*i])
                    x2 = float(splited[3+5*i])
                    y2 = float(splited[4+5*i])
                    c = splited[5+5*i]
                    box.append([x,y,x2,y2])
                    label.append(int(c))
                self._boxes.append(torch.Tensor(box))
                self._labels.append(torch.LongTensor(label))
    
    def _unpack(self,idx):
        fname = self._fnames[idx]
        img = cv2.imread(str(self.root_dir / 'Images' / fname))
        return img,self._boxes[idx],self._labels[idx]

    def __getitem__(self,idx):
        img,boxes,labels=self._unpack(idx)
        img,boxes,labels=self.union_transform(img, boxes, labels)
        return self._encode_target(img,boxes,labels)
    
    def _encode_target(self,img,boxes,labels) -> tuple:
        raise NotImplementedError

    def __len__(self):
        return len(self._boxes)