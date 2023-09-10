import torch
import os
import cv2
from .metric import Metric
from pathlib import Path
import numpy as np
class EvaluatorBase:
    def __init__(self, model:torch.nn.Module,n_classes,color_table, process_image:callable):
        '''
        evaluator base for image semantic segmentation

        model: torch net

        n_classes: class num

        color_tabel: a list of color, i-th is the color of i-th class

        get_img: a method for getting a image, signature: get_img(path:str, *args) -> None

        get_img_args: additional params for get_img
        '''
        self.model=model
        self.n_classes=n_classes
        self.color_table=color_table
        self.process_image=process_image
    
    def test_folder(self,img_dir,folder='origin',destination_folder='masked',finish_hook:callable=None):
        '''
        test multiple images, saving masked image to img_dir/masked/

        img_dir: test folder

        folder: folder in img_dir that contains original images
            
            eg: finish_hook( file_name :str )

        finish_hook: hook after predicted an image
        '''
        _dst_folder=Path(img_dir) / destination_folder
        files=map(
            lambda s:  (s, _dst_folder / s.name),
            Path.glob(Path(img_dir) / folder,'*.*')
        )
        # _tmp=list(files)
        if not Path.exists(_dst_folder):
            Path.mkdir(_dst_folder)

        for f,r in files:
            self.test_img(f,r)
            if finish_hook is not None:
                finish_hook(f)

    def test_img(self,img_path,out_path):
        '''
        test one img, and save colored mask to file
        '''
        img,origin_shape= self.process_image(str(img_path))

        pred_img = self.get_masked_results(img)
        pred_img = cv2.cvtColor(pred_img,cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_path),pred_img)

    def get_masked_results(self,img:torch.Tensor) -> np.ndarray:
        """
        predict one image and render the results in color

        Args:
            img (np.ndarray): original image (pre-processed)

        Returns:
            results picture, (BGR)
        """
        o_c ,o_h, o_w = img.shape

        inputs = torch.unsqueeze(img, 0) # [b,c,h,w]
        with torch.no_grad():
            output = self.model(inputs)

        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        assert (N == 1)
        pred = output.transpose(0, 2, 3, 1).reshape(-1, self.n_classes).argmax(axis=1).reshape(h, w)

        pred_img = np.zeros((h, w, 3), dtype=np.float32)
        for cls in range(self.n_classes):
            pred_inds = pred == cls
            # label = index2label[cls]
            color=self.color_table[cls]
            pred_img[pred_inds] = color

        pred_img = cv2.resize(pred_img,(o_w,o_h))
        return pred_img

    def calculate_metrics(self,pred,gt):
        metric=Metric(self.n_classes)
        metric.add_batch(gt,pred)
        return metric.Pixel_Accuracy(),metric.Mean_Intersection_over_Union()