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
                crop_factor=32, 
                flip_rate=0.5,
                rot_rate=0.5,
                downsize=1,
                json_file='train_val_data.json',
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

        tmp=json.load(open(self.dir / json_file,'r'))
        
        self.train_files = tmp['train']
        self.val_files   = tmp['val']

        self.flip_rate = flip_rate
        self.rot_rate= rot_rate
        if self.train is False:
            self.flip_rate = 0.
            self.rot_rate= 0.
        

        self.crop_factor= crop_factor
        self.downsize= downsize

        self.transform=transform
        self.target_transform=target_transform
        
        if(self.train):
            self.data=self.train_files
        else:
            self.data=self.val_files
    
    def process_image(self,path):
        raise NotImplementedError
    
    def process_label(self,path):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name=self.data[idx]['img']
        label_name=self.data[idx]['label']

        img,_=self.process_image(str(self.dir / img_name))
        label = self.process_label(str(self.dir / label_name))

        # img,label=flip(img,label,self.flip_rate)
        # img,label=rot90(img,label,self.rot_rate)

        # create one-hot encoding     
        target = torch.zeros(self.n_class, *label.shape)
        for c in range(self.n_class):
            target[c][label == c] = 1

        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)

        sample = {'X': img, 'Y': target, 'l': label}

        return sample


def process_image(img: [str | Path | np.ndarray],downsize:int,crop_factor:int,MEAN:[list | tuple], STD:[list | tuple]):
    """
    Pre-procss the image, including

    1. load (only if img is str or Path)
    2. bgr to rgb
    3. to tensor (chw)
    4. downsize
    5. crop
    6. reduce mean

    Args:
        img (str  |  Path  |  np.ndarray]): _description_
        downsize (int): _description_
        crop_factor (int): _description_
        MEAN (list  |  tuple]): _description_
        STD (list  |  tuple]): _description_

    Returns:
        img (torch.Tensor), shape
    """    
    if not isinstance(img,np.ndarray):
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # [h,w,c]

    shape=img.shape
    h, w = (shape[0],shape[1])
    
    # convert to tensor
    img = torch.from_numpy(img.copy()).float()
    # [h,w,c] -> [c,h,w]
    img=img.permute(2,0,1)
    # downsize
    img=img[:,::downsize,::downsize]
    _new_h= ((h//downsize)//crop_factor)*crop_factor
    _new_w= ((w//downsize)//crop_factor)*crop_factor
    # crop
    img=transforms.CenterCrop((_new_h,_new_w))(img)
    # reduce mean
    img=img/255.
    img=transforms.Normalize(MEAN,STD)(img)
    return img,shape

def process_label(path,downsize:int,crop_factor:int):
    '''
    load

    to tensor (chw)

    downsize

    crop
    '''
    label = cv2.imread(path)
    # convert to tensor
    label = torch.from_numpy(label.copy()).long()
    # [h,w,c] -> [c,h,w] -> [h,w]
    label=label.permute(2,0,1)[0]
    h, w = label.shape
    # downsize
    label=label[::downsize,::downsize]
    # crop
    _new_h= ((h//downsize)//crop_factor)*crop_factor
    _new_w= ((w//downsize)//crop_factor)*crop_factor
    label=transforms.CenterCrop((_new_h,_new_w))(label)
    return label

def flip(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        img=torch.flip(img,[-1])
        label=torch.flip(label,[-1])
    return img,label

def rot90(img,label,p):
    '''
    Horizonal Flip at possibility P
    '''
    if(torch.rand(1).item()<p):
        k=torch.randint(1,3,(1,)).item()
        img=torch.rot90(img,k)
        label=torch.rot90(label,k)
    return img,label