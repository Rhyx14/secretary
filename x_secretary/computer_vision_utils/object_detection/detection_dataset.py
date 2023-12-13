import os,sys,random,fileinput,torch,cv2
from pathlib import Path
import torch.utils.data as data
from x_secretary.computer_vision_utils.union_transforms import Union_Transforms
class Dectection_Dataset(data.Dataset):
    def __init__(self,root_dir,file_list,transforms):

        self.root_dir=Path(root_dir)

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.union_transform=Union_Transforms(transforms)

        with fileinput.input(map(lambda f: str(self.root_dir / f),file_list)) as file:
            for line in file:
                if(line == '' ): continue
                # txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
                splited = line.strip().split()
                self.fnames.append(splited[0])
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
                    label.append(int(c)+1)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.LongTensor(label))
    
    def _unpack(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(str(self.root_dir / 'Images' / fname))
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        return img,boxes,labels

    def __getitem__(self,idx):
        img,boxes,labels=self._unpack(idx)

        # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug

        img,boxes,labels=self.union_transform(img, boxes, labels)

        target = self._encode_target(img,boxes,labels)

        return img,target
    
    def _encode_target(self,img,boxes,labels):
        raise NotImplementedError

    def __len__(self):
        return len(self.boxes)