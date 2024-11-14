import torch,accelerate
import torch.distributed
from .pipelinebase import PipelineBase
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
                 dl_workers=4, 
                 prefetch_factor=2, 
                 default_device='cpu', 
                 mode=Image_training.Mode.CLASSIFICATION):
        self.cfg=cfg
        self._ddp=torch.distributed.is_torchelastic_launched()
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
        
        super().__init__(cfg, on_epoch_begin, on_epoch_end, on_turn_begin, on_turn_end, dl_workers, prefetch_factor, default_device, mode)


    def Run(self,mix_precision=False,*args,**kwargs):
        CFG=self.cfg
        CFG.Teacher.eval()
        CFG.Student.train()

        scaler=accelerate.utils.get_grad_scaler()
        for ep in range(CFG.EPOCH):
            
            PipelineBase.call_hooks(self.on_epoch_begin,ep)
            for _b_id,datum in enumerate(self._dl):

                x,y=self._unpack(datum)

                PipelineBase.call_hooks(self.on_turn_begin,_b_id)
                
                for _2_opt in CFG.opt : _2_opt.zero_grad(set_to_none=True)
                
                with self._accelerator.autocast():
                    _loss = CFG.KD_loss(x,y)
                if mix_precision:
                    scaler.scale(_loss).backward()
                    for _2_opt in CFG.opt : scaler.step(_2_opt)
                    scaler.update()

                else:
                    _loss = CFG.KD_loss(x,y)
                    _loss.backward()
                    for _2_opt in CFG.opt: _2_opt.step()
                            
                PipelineBase.call_hooks(self.on_turn_end,len(self._dl),_b_id,_loss.item(),ep)

            if hasattr(CFG,'lr_scheduler'):
                for __lr_sch in CFG.lr_scheduler: __lr_sch.step()

            PipelineBase.call_hooks(self.on_epoch_end,_loss.item(),ep)
        return
