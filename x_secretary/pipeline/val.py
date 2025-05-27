import torch,accelerate
from .pipelinebase import PipelineBase
from .misc import DDP_progressbar
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
class Image_classification_val(PipelineBase):
    '''
    Image evaluation pipline for image classification

    on_turn_begin : hooks before each training turn, with parameter ()

    on_turn_end : hooks after each training turn, with parameter (batch_id)

    mix_precision: Choose from 'no','fp16','bf16' or 'fp8' (achieved via accelerate)
    '''
    def __init__(self,
            batch_size,
            net,
            dataset,
            dl_workers=4,
            dl_prefetch_factor=2,
            on_turn_begin=None,
            on_turn_end=None,
            mix_precision='fp16',
            cpu=False,
            data_hooks=None,
            get_pred=None
        ) -> None:

        self.net=net
        self.batch_size=batch_size
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor
        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end
        self._mix_precision=mix_precision

        if get_pred is not None: self._get_pred=get_pred
        else: self._get_pred=lambda x: x.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        self._accelerator=Accelerator(
            dataloader_config=accelerate.utils.DataLoaderConfiguration(
                split_batches=True,
                even_batches=False
            ),
            mixed_precision=self._mix_precision,
            cpu=cpu)
        
        self._dl=self._accelerator.prepare_data_loader(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.dl_workers,
                prefetch_factor=self.dl_prefetch_factor,
                shuffle=False,
                pin_memory=True)
            )
        
        super().__init__(self._accelerator.device,data_hooks)

    def __call__(self,loss=None,*args,**kwargs):
        with torch.no_grad():
            acc=torch.Tensor([0]).to(self._accelerator.device)
            _loss=torch.Tensor([0]).to(self._accelerator.device)

            for _bid,datum in enumerate(DDP_progressbar(self._dl)):
                x,label=PipelineBase.call_hooks(self.data_hooks,datum)
                PipelineBase.call_actions(self.on_turn_begin)

                # simulate snn
                with self._accelerator.autocast():
                    _out=self.net(x) 

                if loss is not None:
                    _loss = (_loss*_bid + loss(_out,label).item())/(_bid+1)
                
                pred= self._get_pred(_out)
                acc += pred.eq(label.view_as(pred)).sum().item() 
                
                PipelineBase.call_actions(self.on_turn_end,_bid,_loss)

            acc=self._accelerator.reduce(acc,reduction='sum').item()
            _loss=self._accelerator.reduce(_loss,reduction='mean').item()

        r=acc/float(len(self.dataset))
        return acc, r, _loss

class Image_plain_val(Image_classification_val):
    '''
    Image evaluation pipline

    on_turn_begin : hooks before each training turn, with parameter ()

    on_turn_end : hooks after each training turn, with parameter (batch_id)

    mix_precision: Choose from 'no','fp16','bf16' or 'fp8' (achieved via accelerate)
    '''
    def __call__(self,loss=None,*args,**kwargs):
        with torch.no_grad():
            _loss=torch.Tensor([0]).to(self._accelerator.device)

            for _bid,datum in enumerate(DDP_progressbar(self._dl)):
                x,label=PipelineBase.call_hooks(self.data_hooks,datum)
                PipelineBase.call_actions(self.on_turn_begin)
                # simulate
                with self._accelerator.autocast():
                    _out=self.net(x)

                if loss is not None:
                    _loss = (_loss*_bid + loss(_out,label).item())/(_bid+1)

                PipelineBase.call_actions(self.on_turn_end,_bid,_loss)

            _loss=self._accelerator.reduce(_loss,reduction='mean').item()
        return _loss

from ..computer_vision_utils.semantic_segmentation.metric import Metric
class Image_segmentation_val(PipelineBase):
    '''
    val pipline for image semantic segmentation, (solo mode, only on rank 0)

    on_turn_begin : hooks before each training turn, with parameter ()

    on_turn_end : hooks after each training turn, with parameter (batch_id)
    '''
    def __init__(self,
            batch_size,
            n_classes,
            net:torch.nn.Module,
            dataset,
            default_device:str=None,
            dl_workers=4,
            dl_prefetch_factor=2,
            on_turn_begin=None,
            on_turn_end=None,):
        self.net=net
        self.batch_size=batch_size
        self.n_classes=n_classes
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor

        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end
        
        super().__init__(default_device,lambda x: x)

    def __call__(self,*args,**kwargs):

        metric=Metric(self.n_classes)
        dl=DataLoader(self.dataset,
                      batch_size=self.batch_size,
                      num_workers=self.dl_workers,
                      prefetch_factor=self.dl_prefetch_factor,
                      pin_memory=True)
        with torch.no_grad():
            for iter, datum in enumerate(tqdm(dl,leave=False)):
               inputs,gt= PipelineBase.call_hooks(self.data_hooks,datum)

               # simulate snn
               PipelineBase.call_actions(self.on_turn_begin)
               output = self.net(inputs)
               output = output.data.cpu().numpy()

               pred=output.argmax(axis=1)
               
               metric.add_batch(gt,pred)
               PipelineBase.call_actions(self.on_turn_end,iter)

            _acc=metric.Pixel_Accuracy()
            _miou=metric.Mean_Intersection_over_Union()

            return _acc,_miou