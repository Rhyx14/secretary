import torch
import os
import cv2
from pathlib import Path
import numpy as np
import einops
class EvaluatorBase:
    def __init__(self, model:torch.nn.Module,color_table,transforms:callable=None):
        '''
        evaluator base for image semantic segmentation

        model: torch net

        color_tabel: a list of color, i-th is the color of i-th class

        transforms: pre-process of image
        '''
        self.model=model
        self.n_classes=len(color_table)
        self.color_table=color_table
        self.transforms=transforms

    def _load_img(self,img_path):
        img = cv2.imread(str(img_path))
        return img

    def get_pred(self,img:np.ndarray) -> torch.Tensor:
        """
        predict one image

        Args:
            img (np.ndarray): h w c, opencv form

        Returns:
            torch.Tensor: _description_
        """
        if self.transforms is not None:
            img=self.transforms(img)

        with torch.no_grad():
            inputs = torch.unsqueeze(img, 0) # [b c h w]
            output = self.model(inputs).cpu()
            b, c, h, w = output.shape
            assert b == 1 and self.n_classes==c
            pred= einops.rearrange(output,'b c h w -> h w (b c)')
            pred = pred.argmax(dim=2,keepdim=False) # [h w c] -> [h w]

        return pred

    def get_colored_results(self,pred:torch.Tensor) -> np.ndarray:
        """
        predict one image and render the results in color

        Args:
            img (np.ndarray): original image (pre-processed)

        Returns:
            results picture (same as the COLOR table)
        """
        o_h, o_w = pred.shape
        pred = pred.cpu().numpy()
        pred_img = np.zeros((o_h, o_w, 3), dtype=np.uint8)
        for cls in range(self.n_classes):
            pred_inds = pred == cls
            color=self.color_table[cls]
            pred_img[pred_inds] = color

        return pred_img
        
    def test_img_from_file(self,img_path: str,out_path:str):
        '''
        test one img, and save colored mask to file,

        Won't saving if out_path is None

        return (original_image, predicted_image)
        '''
        img= self._load_img(str(img_path))
        shape=img.shape
        pred = self.get_pred(img)
        pred_img = self.get_colored_results(pred)
        pred_img = cv2.resize(pred_img,(shape[1],shape[0])) # w,h
        # pred_img = cv2.cvtColor(pred_img,cv2.COLOR_RGB2BGR)
        if out_path is not None:
            cv2.imwrite(str(out_path),pred_img)
        return img,pred_img

    def test_folder(self,img_dir,folder='origin',destination_folder='masked',finish_hook:callable=None,saving_image=True,video=None):
        '''
        test multiple images, saving masked image to img_dir/masked/

        img_dir: test folder

        folder: folder in img_dir that contains original images
            
            eg: finish_hook( file_name :str )

        finish_hook: hook after predicted an image

        saving_image: whether saving each frame

        video: a dict contains {'fps':24, 'file_name':'result.avi'} for saving videos. No video output if set to None 
        '''
        _dst_folder=Path(img_dir) / destination_folder
        files=map(
            lambda s:  (Path(img_dir)/folder/s, _dst_folder / s),
            os.listdir(Path(img_dir) / folder),
            # Path.glob(Path(img_dir) / folder,'*.*')
        )
        files=sorted(list(files),key=lambda x:str(x[0]))
        if not Path.exists(_dst_folder):
            Path.mkdir(_dst_folder)

        rslt=[]
        for f,r in files:
            rslt.append(self.test_img_from_file(f,r if saving_image else None))
            if finish_hook is not None: finish_hook(f)

        if video is not None:
            video={'fps':24, 'file_name':'result.avi'} | video
            if len(rslt)==0: return
            shape=rslt[0][0].shape
            out_stream=cv2.VideoWriter(str(_dst_folder/video['file_name']),cv2.VideoWriter.fourcc(*'MJPG'),video['fps'],(shape[1]*2,shape[0]))
            for _img,_rslt in rslt:
                _out=np.zeros([shape[0],shape[1]*2,shape[2]],dtype=np.uint8)
                _out[:,0:shape[1],:]=_img
                _out[:,shape[1]:,:]=_rslt
                out_stream.write(_out)
            out_stream.release()

    # def calculate_metrics(self,pred,gt):
    #     metric=Metric(self.n_classes)
    #     metric.add_batch(gt,pred)
    #     return metric.Pixel_Accuracy(),metric.Mean_Intersection_over_Union()