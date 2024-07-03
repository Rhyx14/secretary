import torch
from .pipelinebase import PipelineBase,DDP_progressbar
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
class Image_classification_val(PipelineBase):
    '''
    val pipline for image classification

    on_turn_begin : hooks before each training turn, with parameter ()

    on_turn_end : hooks after each training turn, with parameter (batch_id)
    '''
    def __init__(self, 
            logger,
            batch_size,
            net,
            dataset,
            dl_workers=4,
            dl_prefetch_factor=2,
            on_turn_begin=None,
            on_turn_end=None,
        ) -> None:
        super().__init__(None)
        self.net=net
        self.logger=logger
        self.batch_size=batch_size
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor
        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end
        
        from accelerate import Accelerator
        self._accelerator=Accelerator(split_batches=True,even_batches=False)
        self._dl=self._accelerator.prepare_data_loader(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.dl_workers,
                prefetch_factor=self.dl_prefetch_factor,
                shuffle=False,
                pin_memory=True)
            )

    def Run(self,loss=None,mix_precision=False,*args,**kwargs):
        with torch.no_grad():
            acc=torch.Tensor([0]).to(self._accelerator.device)
            _loss=torch.Tensor([0]).to(self._accelerator.device)

            for _bid,(x,label) in enumerate(DDP_progressbar(self._dl)):

                x=x.to(self._accelerator.device,non_blocking=True)
                label=label.to(self._accelerator.device,non_blocking=True)
                PipelineBase.call_hooks(self.on_turn_begin)

                # simulate snn
                if(mix_precision):
                    with autocast():
                        _out=self.net(x)
                else:
                    _out=self.net(x)  

                if loss is not None:
                    _loss = (_loss*_bid + loss(_out,label).item())/(_bid+1)

                pred = _out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                acc += pred.eq(label.view_as(pred)).sum().item() 
                
                PipelineBase.call_hooks(self.on_turn_end,_bid,_loss)

            acc=self._accelerator.reduce(acc,reduction='sum').item()
            _loss=self._accelerator.reduce(_loss,reduction='mean').item()

        r=acc/float(len(self.dataset))
        return acc, r, _loss

class Image_plain_val(Image_classification_val):
    def __init__(self, 
            logger,
            batch_size,
            net,
            dataset,
            dl_workers=4,
            dl_prefetch_factor=2,
            on_turn_begin=None,
            on_turn_end=None,
        ) -> None:
        super().__init__(logger,batch_size,net,dataset,dl_workers,dl_prefetch_factor,on_turn_begin,on_turn_end)

    def Run(self,loss=None,mix_precision=False,*args,**kwargs):
        with torch.no_grad():
            _loss=torch.Tensor([0]).to(self._accelerator.device)

            for _bid,(x,label) in enumerate(DDP_progressbar(self._dl)):
                x=x.to(self._accelerator.device,non_blocking=True)
                label=label.to(self._accelerator.device,non_blocking=True)

                PipelineBase.call_hooks(self.on_turn_begin)
                # simulate
                if(mix_precision):
                    with autocast():
                        _out=self.net(x)
                else:
                    _out=self.net(x)  

                if loss is not None:
                    _loss = (_loss*_bid + loss(_out,label).item())/(_bid+1)

                PipelineBase.call_hooks(self.on_turn_end,_bid,_loss)

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
            logger,
            batch_size,
            n_classes,
            net:torch.nn.Module,
            dataset,
            cuda_device:str=None,
            dl_workers=4,
            dl_prefetch_factor=2,
            on_turn_begin=None,
            on_turn_end=None,):
        super().__init__()
        self.net=net
        self.logger=logger
        self.batch_size=batch_size
        self.cuda_device=cuda_device
        self.n_classes=n_classes
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor

        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end

    def Run(self,*args,**kwargs):

        metric=Metric(self.n_classes)
        dl=DataLoader(self.dataset,
                      batch_size=self.batch_size,
                      num_workers=self.dl_workers,
                      prefetch_factor=self.dl_prefetch_factor,
                      pin_memory=True)
        with torch.no_grad():
            for iter, datum in enumerate(tqdm(dl,leave=False)):
               inputs = datum['X'].to(self.cuda_device)
               gt=datum['Y'].cpu().numpy()

               # simulate snn
               PipelineBase.call_hooks(self.on_turn_begin)
               output = self.net(inputs)
               output = output.data.cpu().numpy()

               pred=output.argmax(axis=1)
               
               metric.add_batch(gt,pred)
               PipelineBase.call_hooks(self.on_turn_end,iter)

            _acc=metric.Pixel_Accuracy()
            _miou=metric.Mean_Intersection_over_Union()

            return _acc,_miou