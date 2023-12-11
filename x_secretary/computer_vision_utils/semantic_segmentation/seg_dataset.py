from torch.utils.data import Dataset
from pathlib import Path
import json,torch,cv2
from ..union_transforms import Union_Transforms
import numpy as np
class Seg_Dataset(Dataset):
    def __init__(self, 
                dir,
                n_class,
                train,
                json_file,
                transforms:list):
        """
        Dataset for semantic segmentation task only

        Args:
            dir (_type_): the dataset path, the dataset contains:
                1. original images
                2. mask images (gray scale, the pixel represents the category index)
                3. a json file contains the path of each sample, such as 
                    {
                        'train':[
                            datum // any obj,
                            datum // any obj,
                            ...
                        ],
                        'val:':[
                            datum // any obj,
                            ...
                        ]
                    }
                The inner dict may varies from different datasets. It's always necessitates an override of method: self._unpack_img_label(self,datum).
                The default method suppose the datum in the formate as:
                    ['relative/str/path/to/image','relative/str/path/to/label']

            n_class (_type_): number of classes,
            train (bool, optional): train / val mode. Defaults to False.
            crop_factor (int, optional): align the width and height. Defaults to 32.
            flip_rate (float, optional): augmentation, horizontal flip. Defaults to 0.5.
            rot_rate (float, optional): augmentation, rotate 90 degree. Defaults to 0.5.
            downsize (int, optional): down size factor. Defaults to 1.
            json_file (str, optional): json file path that contain data path.

            union_transform (list, optional): transform. Defaults to None. (img,label) -> img, label

            The squence is: union_transform -> (target) transform.
        """         
        super(Seg_Dataset,self).__init__()
        self.train=train
        self.dir=Path(dir)
        self.n_class=n_class

        tmp=json.loads(Path.read_text(self.dir / json_file))
        
        self.train_files = tmp['train']
        self.val_files   = tmp['val']

        self.union_transform=Union_Transforms(transforms)
        
        if(self.train):
            self.data=self.train_files
        else:
            self.data=self.val_files

    def __len__(self):
        return len(self.data)

    def _unpack_img_label(self,datum):
        img = cv2.imread(str(self.dir / datum[0]))
        label= cv2.imread(str(self.dir / datum[1]))
        return img,label

    def __getitem__(self, idx):
        img,label=self._unpack_img_label(self.data[idx])

        img,label=self.union_transform(img,label)

        # create one-hot encoding     
        target = torch.zeros(self.n_class, *label.shape)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label}

        return sample

