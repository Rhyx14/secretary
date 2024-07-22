import torch
from .pipelinebase import PipelineBase
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast
from .train import Image_training
class Image_KD_training(Image_training):
    '''
    training pipeline for pytorch distribution mode,

    cfg should contains following keys:

    {
    
        'train_dataset':torch.data.Dataset,

        'Teacher':torch.nn.Module  | torch.nn.parallel.distributed.DistributedDataParallel,

        'Student':torch.nn.Module  | torch.nn.parallel.distributed.DistributedDataParallel,

        'KD_loss':torch.nn.Module,

        'opt':torch.optim.Optimizer,

        'BATCH_SIZE':int,

        'EPOCH':int

    }

    ---------
    on_epoch_begin : hooks before each epoch, with parameter (ep)

    on_epoch_end : hooks after each epoch, with parameter (loss,ep)

    on_turn_begin : hooks before each training turn, with parameter (ep, batch_id)

    on_turn_end : hooks after each training turn, with parameter (batch len,batch_id,loss,ep)
    '''
    def __init__(self, 
                 cfg, 
                 on_epoch_begin=None, 
                 on_epoch_end=None, 
                 on_turn_begin=None, 
                 on_turn_end=None, 
                 DDP=False, 
                 dl_workers=4, 
                 prefetch_factor=2, 
                 default_device='cpu', 
                 mode=Image_training.Mode.CLASSIFICATION):
        self.cfg=cfg
        self._ddp=DDP
        self.loss=None # only for the attribute check in the super class
        PipelineBase._Check_Attribute(self.cfg,'Teacher',torch.nn.Module)
        if self._ddp:
            PipelineBase._Check_Attribute(self.cfg,'Student',torch.nn.parallel.distributed.DistributedDataParallel)
            cfg.net=torch.nn.parallel.distributed.DistributedDataParallel(torch.nn.Module()) # only for the attribute check in the super class
            cfg.loss=lambda x: x
        else:
            PipelineBase._Check_Attribute(self.cfg,'Student',torch.nn.Module)
            cfg.net=torch.nn.Module() # only for the attribute check in the super class
            cfg.loss=lambda x: x
        PipelineBase._Check_Attribute(self.cfg,'KD_loss',(object,)) 
        
        super().__init__(cfg, on_epoch_begin, on_epoch_end, on_turn_begin, on_turn_end, DDP, dl_workers, prefetch_factor, default_device, mode)


    def Run(self,mix_precision=False,*args,**kwargs):
        CFG=self.cfg
        CFG.Teacher.eval()
        CFG.Student.train()

        scaler=GradScaler()
        for ep in range(CFG.EPOCH):
            
            PipelineBase.call_hooks(self.on_epoch_begin,ep)
            for _b_id,datum in enumerate(self._dl):

                x,y=self._unpack(datum)

                PipelineBase.call_hooks(self.on_turn_begin,_b_id)
                
                if CFG.opt is not list: CFG.opt.zero_grad(set_to_none=True)
                else:
                    for __opt in CFG.opt : __opt.zero_grad(set_to_none=True)
                
                if mix_precision:
                    with autocast():
                        _loss = CFG.KD_loss(x,y)
                    scaler.scale(_loss).backward()
                    if CFG.opt is not list: scaler.step(CFG.opt)
                    else:
                        for __opt in CFG.opt : scaler.step(__opt)
                    scaler.update()

                else:
                    _loss = CFG.KD_loss(x,y)
                    _loss.backward()
                    if CFG.opt is not list: CFG.opt.step()
                    else:
                        for __opt in CFG.opt: __opt.step()
                            
                PipelineBase.call_hooks(self.on_turn_end,len(self._dl),_b_id,_loss.item(),ep)

            if hasattr(CFG,'lr_scheduler'):
                if CFG.lr_scheduler is not list: CFG.lr_scheduler.step()
                else:
                    for __lr_sch in CFG.lr_scheduler: __lr_sch.step()

            PipelineBase.call_hooks(self.on_epoch_end,_loss.item(),ep)
        return
