from enum import Enum
import torch
from torch.utils.data.dataloader import DataLoader
from ..utils.ddp_sampler import DDP_BatchSampler
from .pipeline_base import PipelineBase
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
class Image_training(PipelineBase):
    '''
    training pipeline for pytorch distribution mode,
    
    -----

    cfg should contains following keys:

    {
    
        'train_dataset':torch.data.Dataset,

        'net':torch.nn.Module,

        'loss':torch.nn.Module,

        'opt':torch.optim.Optimizer,

        'BATCH_SIZE':int,

        'EPOCH':int

    }

    ---------
    before_epoch_hooks : hooks before each epoch, with parameter (configuration)

    after_epoch_hooks : hooks after each epoch, with parameter (configuration,loss,ep)

    before_turn_hooks : hooks before each training turn, with parameter (configuration)

    after_turn_hooks : hooks after each training turn, with parameter (configuration,batch_id,loss,ep)

    mode: see Image_training2.Mode
    '''
    class Mode(Enum):
        CLASSIFICATION=1
        SEGMENTATION=2

    def __init__(self,
        cfg,
        before_epoch_hooks=None,
        after_epoch_hooks=None,
        before_turn_hooks=None,
        after_turn_hooks=None,
        ddp=False,
        dl_workers=3,prefetch_factor=2,
        default_device='cpu',
        mode='classification'
        ):
        self.cfg=cfg
        self.ddp=ddp
        self.after_epoch_hooks=after_epoch_hooks
        self.before_epoch_hooks=before_epoch_hooks
        self.after_turn_hooks=after_turn_hooks
        self.before_turn_hooks=before_turn_hooks

        if self.ddp:
            PipelineBase._Check_Attribute(self.cfg,'net',torch.nn.parallel.distributed.DistributedDataParallel)
        else:
            PipelineBase._Check_Attribute(self.cfg,'net',torch.nn.Module)
        PipelineBase._Check_Attribute(self.cfg,'loss')
        PipelineBase._Check_Attribute(self.cfg,'opt',(torch.optim.Optimizer,))
        PipelineBase._Check_Attribute(self.cfg,'train_dataset',(torch.utils.data.Dataset,))
        PipelineBase._Check_Attribute(self.cfg,'BATCH_SIZE',(int,))
        PipelineBase._Check_Attribute(self.cfg,'EPOCH',(int,))

        CFG=self.cfg
        if self.ddp:
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
                prefetch_factor=prefetch_factor,
                shuffle=True,
                drop_last=True)
        
        match mode:
            case Image_training.Mode.CLASSIFICATION: self._unpack = self._unpack_cls
            case Image_training.Mode.SEGMENTATION  : self._unpack = self._unpack_seg
            case _ : raise  NotImplementedError(f'Pipleline for {mode} hasn''t been implemented yet.')

        super().__init__(default_device)

    def Run(self,mix_precision=False,*args,**kwargs):
        CFG=self.cfg

        CFG.net.train()
        
        scaler=GradScaler()

        for ep in range(CFG.EPOCH):
            
            PipelineBase.call_hooks(self.before_epoch_hooks,self.cfg)
            for _b_id,datum in enumerate(self._dl):

                x,y=self._unpack(datum)

                PipelineBase.call_hooks(self.before_turn_hooks,self.cfg)

                if(mix_precision):
                    with autocast():
                        _out=CFG.net(x)
                        _loss = CFG.loss(_out,y)
                    scaler.scale(_loss).backward()
                else:
                    _out=CFG.net(x)
                    _loss = CFG.loss(_out,y)
                    _loss.backward()

                CFG.opt.zero_grad(set_to_none=True)                   
                PipelineBase.call_hooks(self.after_turn_hooks,self.cfg,_b_id,_loss.item(),ep)

            if hasattr(CFG,'lr_scheduler'):
                CFG.lr_scheduler.step()

            PipelineBase.call_hooks(self.after_epoch_hooks,self.cfg,_loss.item(),ep)
        return
