from enum import Enum
import torch
from torch.utils.data.dataloader import DataLoader
from ..utils.ddp_sampler import DDP_BatchSampler
from .pipeline_base import PipelineBase
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
class Image_training(PipelineBase):
    '''
    A training pipeline for supervised learning,
    
    -----

    cfg should contains following keys:

    {
    
        'train_dataset':torch.data.Dataset,

        'net':torch.nn.Module  | torch.nn.parallel.distributed.DistributedDataParallel,

        'loss':torch.nn.Module,

        'opt':torch.optim.Optimizer,

        'BATCH_SIZE':int,

        'EPOCH':int

    }

    ---------
    on_epoch_begin : hooks before each epoch, with parameter (ep)

    on_epoch_end : hooks after each epoch, with parameter (loss,ep)

    on_turn_begin : hooks before each training turn, with parameter (ep, batch_id)

    on_turn_end : hooks after each training turn, with parameter (batch len,batch_id,loss,ep)

    mode: see Image_training2.Mode
    '''
    class Mode(Enum):
        CLASSIFICATION=1
        SEGMENTATION=2
        YOLO_DETECTION=3

    def __init__(self,
        cfg,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_turn_begin=None,
        on_turn_end=None,
        DDP=False,
        dl_workers=4,prefetch_factor=2,
        default_device='cpu',
        mode=Mode.CLASSIFICATION
        ):
        self._cfg=cfg
        self._ddp=DDP
        self.on_epoch_end=on_epoch_end
        self.on_epoch_begin=on_epoch_begin
        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end

        if self._ddp:
            PipelineBase._Check_Attribute(self._cfg,'net',torch.nn.parallel.distributed.DistributedDataParallel)
        else:
            PipelineBase._Check_Attribute(self._cfg,'net',torch.nn.Module)
        PipelineBase._Check_Attribute(self._cfg,'loss')
        PipelineBase._Check_Attribute(self._cfg,'opt',(torch.optim.Optimizer,))
        PipelineBase._Check_Attribute(self._cfg,'train_dataset',(torch.utils.data.Dataset,))
        PipelineBase._Check_Attribute(self._cfg,'BATCH_SIZE',(int,))
        PipelineBase._Check_Attribute(self._cfg,'EPOCH',(int,))

        CFG=self._cfg
        if self._ddp:
            self._dl=DataLoader(CFG.train_dataset,
                num_workers=dl_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                batch_sampler=DDP_BatchSampler(
                    CFG.train_dataset,
                    shuffle=True,
                    batch_size=CFG.BATCH_SIZE))
        else:
            self._dl=DataLoader(CFG.train_dataset,
                batch_size=CFG.BATCH_SIZE,
                num_workers=dl_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                shuffle=True,
                drop_last=True)
        
        match mode:
            case Image_training.Mode.CLASSIFICATION: self._unpack = self._unpack_cls
            case Image_training.Mode.SEGMENTATION  : self._unpack = self._unpack_seg
            case Image_training.Mode.YOLO_DETECTION: self._unpack = self._unpack_cls
            case _ : raise  NotImplementedError(f'Pipleline for {mode} hasn''t been implemented yet.')

        super().__init__(default_device)

    def Run(self,mix_precision=False,*args,**kwargs):
        CFG=self._cfg

        CFG.net.train()
        
        scaler=GradScaler()
        _batch_len=len(self._dl)
        for ep in range(CFG.EPOCH):
            
            PipelineBase.call_hooks(self.on_epoch_begin,ep)
            for _b_id,datum in enumerate(self._dl):

                x,y=self._unpack(datum)

                PipelineBase.call_hooks(self.on_turn_begin,ep,_b_id)
                CFG.opt.zero_grad(set_to_none=True)

                if(mix_precision):
                    with autocast():
                        _out=CFG.net(x)
                        _loss = CFG.loss(_out,y)
                    scaler.scale(_loss).backward()
                    scaler.step(CFG.opt)
                    scaler.update()
                    
                else:
                    _out=CFG.net(x)
                    _loss = CFG.loss(_out,y)
                    _loss.backward()
                    CFG.opt.step()
                PipelineBase.call_hooks(self.on_turn_end,_batch_len,_b_id,_loss.item(),ep)

            if hasattr(CFG,'lr_scheduler'):
                CFG.lr_scheduler.step()

            PipelineBase.call_hooks(self.on_epoch_end,_loss.item(),ep)
        return

from typing import Any
from data_recorder import Avg
class Record_Loss:
    def __init__(self,secretary,name_prefix='') -> None:
        self._secretary=secretary
        self._name_prefix=name_prefix
        pass

    def __call__(self, batch_len,batch_id,loss_value,epochs) -> Any:
        _key=Avg(f'{self._name_prefix}_training_loss_{epochs}',step=batch_id)
        self._secretary.data[_key]=loss_value
        self._secretary.print_solo(f'Epoch {epochs}: {batch_id}/{batch_len}, training loss: {loss_value:.5f} - avg: {self._secretary.data[_key]:.10f}',end='\r')