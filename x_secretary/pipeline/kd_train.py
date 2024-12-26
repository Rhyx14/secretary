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
    the pipeline will pass training_status:dict to each actions, which contains keys:

    batch_len: the number of mini-batches in one epoch

    ep: the index of current epoch

    batch_id: the index of mini-batch in this epoch

    loss: the loss value (value not torch.Tensor)

    '''
    def __init__(self, 
                 cfg, 
                 on_epoch_begin=None, 
                 on_epoch_end=None, 
                 on_turn_begin=None, 
                 on_turn_end=None, 
                 dl_workers=4, 
                 prefetch_factor=2, 
                 mixed_precision='fp16',
                 default_device='cpu', 
                 data_hooks=None):
        self.cfg=cfg
        self._ddp=torch.distributed.is_torchelastic_launched()
        self.loss=None # only for the attribute check in the super class
        PipelineBase._Check_Attribute(self.cfg,'Teacher',torch.nn.Module)
        if self._ddp:
            PipelineBase._Check_Attribute(self.cfg,'Student',torch.nn.parallel.distributed.DistributedDataParallel)
            cfg.net=cfg.Student # only for the attribute check in the super class
            cfg.loss=lambda x: x
        else:
            PipelineBase._Check_Attribute(self.cfg,'Student',torch.nn.Module)
            cfg.net=cfg.Student # only for the attribute check in the super class
            cfg.loss=lambda x: x
        PipelineBase._Check_Attribute(self.cfg,'KD_loss',(object,)) 
        
        super().__init__(cfg, on_epoch_begin, on_epoch_end, on_turn_begin, on_turn_end, dl_workers, prefetch_factor, mixed_precision,default_device, data_hooks)


    def _get_loss(self, datum, CFG):
        with self._accelerator.autocast():
            _loss = CFG.KD_loss(datum[0],datum[1])
        return _loss