import torch
from .pipeline_base import PipelineBase
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast
class Image_Val_Pipeline_ddp(PipelineBase):
    def __init__(self, 
            logger,
            batch_size,
            net,
            dataset,
            dl_workers=4,
            dl_prefetch_factor=4,
            before_turn_hooks=None,
            after_turn_hooks=None,
        ) -> None:
        super().__init__(logger, net, None, None)
        self.batch_size=batch_size
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor
        self.before_turn_hooks=before_turn_hooks
        self.after_turn_hooks=after_turn_hooks
        
        from accelerate import Accelerator
        self._accelerator=Accelerator(split_batches=True,even_batches=False)
        self._dl=self._accelerator.prepare_data_loader(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.dl_workers,
                prefetch_factor=self.dl_prefetch_factor,
                # shuffle=True,
                pin_memory=True)
            )

    def Run(self,loss=None,mix_precision=False,*args,**kwargs):
        with torch.no_grad():
            acc=torch.Tensor([0]).to(self._accelerator.device)
            _loss=torch.Tensor([0]).to(self._accelerator.device)

            for _bid,(x,label) in enumerate(self._dl):

                # x=x.to(self._accelerator.device,non_blocking=True)
                # label=label.to(self._accelerator.device,non_blocking=True)
                PipelineBase.call_hooks(self.before_turn_hooks)

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
                
                PipelineBase.call_hooks(self.after_turn_hooks,_bid,_loss)

            acc=self._accelerator.reduce(acc,reduction='sum').item()
            _loss=self._accelerator.reduce(_loss,reduction='mean').item()

        r=acc/float(len(self.dataset))
        return acc, r, _loss

        
