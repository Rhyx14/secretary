from enum import Enum
import torch
import torch.distributed
from torch.utils.data.dataloader import DataLoader
from .pipelinebase import PipelineBase
from ..utils.set_seeds import seed_worker,get_generator
import accelerate
class Image_training(PipelineBase):
    '''
    A training pipeline for supervised learning,
    
    -----

    cfg should contains following keys:

    {
    
        'train_dataset':torch.data.Dataset,

        'net':torch.nn.Module | torch.nn.parallel.distributed.DistributedDataParallel,

        'loss':torch.nn.Module,

        'opt':list of torch.optim.Optimizer,

        'BATCH_SIZE':int,

        'EPOCH':int

    }

    ---------
    on_epoch_begin : hooks before each epoch, with parameter (ep)

    on_epoch_end : hooks after each epoch, with parameter (loss,ep)

    on_turn_begin : hooks before each training turn, with parameter (ep, batch_id)

    on_turn_end : hooks after each training turn, with parameter (batch len,batch_id,loss,ep)

    mix_precision: Choose from 'no','fp16','bf16' or 'fp8', achieved via accelerate

    mode: see Image_training.Mode
    '''
    class Mode(Enum):
        '''
        CLASSIFICATION: for classsification

        CLASSIFICATION_GPU: for classsification, with gpu transforms (require self.transforms)

        SEGMENTATION: for segmentation

        YOLO_DETECTION: for yolov1 based detection
        '''
        CLASSIFICATION=1
        CLASSIFICATION_GPU=2
        SEGMENTATION=3
        YOLO_DETECTION=4

    def __init__(self,
        cfg,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_turn_begin=None,
        on_turn_end=None,
        dl_workers=4,prefetch_factor=2,
        mixed_precision='fp16',
        default_device='cpu',
        mode=Mode.CLASSIFICATION,
        extra_transforms=lambda x:x
        ):

        self._cfg=cfg
        self._ddp=torch.distributed.is_torchelastic_launched()
        self.on_epoch_end=on_epoch_end
        self.on_epoch_begin=on_epoch_begin
        self.on_turn_begin=on_turn_begin
        self.on_turn_end=on_turn_end
        CFG=self._cfg

        if self._ddp:
            PipelineBase._Check_Attribute(self._cfg,'net',torch.nn.parallel.distributed.DistributedDataParallel)
        else:
            PipelineBase._Check_Attribute(self._cfg,'net',torch.nn.Module)
        PipelineBase._Check_Attribute(self._cfg,'loss')
        PipelineBase._Check_Attribute(self._cfg,'opt',(list,))
        PipelineBase._Check_Attribute(self._cfg,'train_dataset',(torch.utils.data.Dataset,))
        PipelineBase._Check_Attribute(self._cfg,'BATCH_SIZE',(int,))
        PipelineBase._Check_Attribute(self._cfg,'EPOCH',(int,))

        self._accelerator=accelerate.Accelerator(
            mixed_precision=mixed_precision,
            dataloader_config=accelerate.utils.DataLoaderConfiguration(
                split_batches=True,data_seed=0,use_seedable_sampler=False
            ))
        self._dl=self._accelerator.prepare_data_loader(
            DataLoader(CFG.train_dataset,
                batch_size=CFG.BATCH_SIZE,
                num_workers=dl_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=get_generator(),
                drop_last=True)
        )
        super().__init__(default_device,extra_transforms)

        match mode:
            case Image_training.Mode.CLASSIFICATION: self._unpack = self._unpack_cls
            case Image_training.Mode.SEGMENTATION  : self._unpack = self._unpack_seg
            case Image_training.Mode.YOLO_DETECTION: self._unpack = self._unpack_cls
            case _ : raise  NotImplementedError(f'Pipleline for {mode} hasn''t been implemented yet.')

    def Run(self,*args,**kwargs):
        CFG=self._cfg

        CFG.net.train()
        
        scaler=accelerate.utils.get_grad_scaler()
        _batch_len=len(self._dl)
        for ep in range(CFG.EPOCH):
            
            PipelineBase.call_hooks(self.on_epoch_begin,ep)
            for _b_id,datum in enumerate(self._dl):

                x,y=self._unpack(datum)
                PipelineBase.call_hooks(self.on_turn_begin,ep,_b_id)

                for __opt in CFG.opt : __opt.zero_grad(set_to_none=True)

                with self._accelerator.autocast():
                    _out=CFG.net(x)
                    _loss = CFG.loss(_out,y)
                    
                    if self._accelerator.mixed_precision is None:
                        _loss.backward()
                        for __opt in CFG.opt : __opt.step()
                    else:
                        scaler.scale(_loss).backward()
                        for __opt in CFG.opt : scaler.step(__opt)
                        scaler.update()                 

                PipelineBase.call_hooks(self.on_turn_end,_batch_len,_b_id,_loss.item(),ep)

            if hasattr(CFG,'lr_scheduler'):
                for __lr_sch in CFG.lr_scheduler: __lr_sch.step()

            PipelineBase.call_hooks(self.on_epoch_end,_loss.item(),ep)
        return
