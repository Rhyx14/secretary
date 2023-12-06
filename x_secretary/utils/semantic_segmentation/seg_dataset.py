from torch.utils.data import Dataset
from pathlib import Path
import json,torch,cv2
from torchvision import transforms
import numpy as np

class Seg_Dataset(Dataset):
    def __init__(self, 
                dir,
                n_class,
                train=False,
                json_file='train_val_data.json',
                union_transform=None,
                transform=None,
                target_transform=None):
        """
        Dataset for semantic segmentation task only

        Args:
            dir (_type_): the dataset path, the dataset contains:
                1. original images
                2. mask images (gray scale, the pixel represents the category index)
                3. a json file contains the path of each sample, such as 
                    {
                        'train':[
                            {'img': path/to/img, 'label': path/to/label},
                            {'img': path/to/img, 'label': path/to/label},
                            ...
                        ],
                        'val:':[
                            {'img': path/to/img, 'label': path/to/label},
                            ...
                        ]
                    }

            n_class (_type_): number of classes,
            train (bool, optional): train / val mode. Defaults to False.
            crop_factor (int, optional): align the width and height. Defaults to 32.
            flip_rate (float, optional): augmentation, horizontal flip. Defaults to 0.5.
            rot_rate (float, optional): augmentation, rotate 90 degree. Defaults to 0.5.
            downsize (int, optional): down size factor. Defaults to 1.
            json_file (str, optional): json file path that contain data path.
            transform (_type_, optional): extra transform. Defaults to None.
            target_transform (_type_, optional): extra transform. Defaults to None.
        """        
        super(Seg_Dataset,self).__init__()
        self.train=train
        self.dir=Path(dir)
        self.n_class=n_class

        tmp=json.load(Path.read_text(self.dir / json_file))
        
        self.train_files = tmp['train']
        self.val_files   = tmp['val']

        self.union_transform=union_transform
        self.transform=transform
        self.target_transform=target_transform
        
        if(self.train):
            self.data=self.train_files
        else:
            self.data=self.val_files

    def _load_img(self,img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # [h,w,c]
        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        # [h,w,c] -> [c,h,w]
        img=img.permute(2,0,1)
        img=img/255.
        return img
    
    def _load_label(self,label_path):
        label = cv2.imread(str(label_path))
        # convert to tensor
        label = torch.from_numpy(label.copy()).long()
        # [h,w,c] -> [c,h,w] -> [h,w]
        label=label.permute(2,0,1)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name=self.data[idx]['img']
        label_name=self.data[idx]['label']

        img= self._load_img(self.dir / img_name)
        label=self._load_label(self.dir/ label_name)

        if self.union_transform is not None:
            if isinstance(self.union_transform,list):
                for _trans in self.union_transform:
                    img,label=_trans(img,label)
                else: 
                    img,label=self.union_transform(img,label)

        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            label=self.target_transform(label)

        # create one-hot encoding     
        target = torch.zeros(self.n_class, *label.shape)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

