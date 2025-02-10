from enum import Enum
import torch
import torch.distributed
from torch.utils.data.dataloader import DataLoader
from .pipelinebase import PipelineBase
from ..utils.set_seeds import seed_worker,get_generator

from .empty_warmup_LR_Scheduler import EmptyWarmup,EmptyLRScheduler
from ..utils.larc import LARC

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

    the pipeline will pass training_status:dict to each actions, which contains keys:

    batch_len: the number of mini-batches in one epoch

    ep: the index of current epoch

    batch_id: the index of mini-batch in this epoch

    loss: the loss value (value not torch.Tensor)

    ---------
    on_epoch_begin : actions before each epoch, with parameter

    on_epoch_end : actions after each epoch, with parameter

    on_turn_begin : actions before each training turn, with parameter

    on_turn_end : actions after each training turn, with parameter

    mix_precision: Choose from 'no','fp16','bf16' or 'fp8', achieved via accelerate

    mode: see Image_training.Mode
    '''
    def __init__(self,
        cfg,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_turn_begin=None,
        on_turn_end=None,
        dl_workers=4,prefetch_factor=2,
        mixed_precision='fp16',
        default_device='cpu',
        data_hooks=None
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
        PipelineBase._Check_Attribute(self._cfg,'opt',(list,torch.optim.Optimizer,LARC))
        PipelineBase._Check_Attribute(self._cfg,'train_dataset',(torch.utils.data.Dataset,))
        PipelineBase._Check_Attribute(self._cfg,'BATCH_SIZE',(int,))
        PipelineBase._Check_Attribute(self._cfg,'EPOCH',(int,))

        self._init_opt_lr_scheduler()

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
        super().__init__(default_device,data_hooks)

    def _init_opt_lr_scheduler(self):
        '''
        If there is a single opt, pack it into a list.

        If lr_scheduler and warmup_scheduler is not designated, an empty scheduler as a placeholder will be set 
        '''
        if not isinstance(self._cfg.opt,list):
            self.opt=[self._cfg.opt]
            self._opt_enable_list=[True]
        else:
            self.opt=self._cfg.opt
            self._opt_enable_list=[True for _ in range(len(self.opt))]
            
        if hasattr(self._cfg,'lr_scheduler'):
            if isinstance(self._cfg.lr_scheduler,list):
                self._lr_scheduler=self._cfg.lr_scheduler
            else:
                self._lr_scheduler=[self._cfg.lr_scheduler]
        else:
            self._lr_scheduler=[EmptyLRScheduler()]
            
        if hasattr(self._cfg,'warmup_scheduler'):
            if isinstance(self._cfg.warmup_scheduler,list):
                self._warmup_scheduler=self._cfg.warmup_scheduler
            else:
                self._warmup_scheduler=[self._cfg.warmup_scheduler]
        else:
            self._warmup_scheduler=[EmptyWarmup()]

    def _zero_grad_opt(self):
        for _opt,_enable_flag in zip(self.opt,self._opt_enable_list):
            if _enable_flag:
                _opt.zero_grad(set_to_none=True)

    def _step_opt(self,scaler=None):
        for _opt,_enable_flag in zip(self.opt,self._opt_enable_list):
            if _enable_flag:
                if scaler == None:
                    _opt.step()
                else:
                    scaler.step(_opt)


    def enable_opt(self,list : list[bool]):
        '''
        Enable optimizer during training, 'True' marks enabled
        '''
        self._opt_enable_list=list

    def _get_loss(self,datum,CFG):
        '''
        Get the final loss tensor for back-propagation
        '''
        with self._accelerator.autocast():
            _out=CFG.net(datum[0])
            _loss = CFG.loss(_out,datum[1])
        return _loss

    def Run(self,early_stop=100000,*args,**kwargs):
        CFG=self._cfg
        
        scaler=accelerate.utils.get_grad_scaler()
        batch_len=len(self._dl)
        training_status={'batch_len':batch_len,'pipline_object':self}
        for _ep in range(CFG.EPOCH):
            if _ep>early_stop: break
            training_status['ep']=_ep
            PipelineBase.call_actions(self.on_epoch_begin,training_status)
            for _b_id,datum in enumerate(self._dl):
                
                training_status['batch_id']=_b_id
                datum=PipelineBase.call_hooks(self.data_hooks,datum)
                PipelineBase.call_actions(self.on_turn_begin,training_status)

                self._zero_grad_opt()

                _loss=self._get_loss(datum,CFG)
                    
                if self._accelerator.mixed_precision is None:
                    _loss.backward()
                    self._step_opt()
                else:
                    scaler.scale(_loss).backward()
                    self._step_opt(scaler)
                    scaler.update()

                training_status['loss']=_loss.item()
                PipelineBase.call_actions(self.on_turn_end,training_status)

            for _lr_sch,_warmup_sch in zip(self._lr_scheduler,self._warmup_scheduler):
                with _warmup_sch.dampening():
                    _lr_sch.step()

            PipelineBase.call_actions(self.on_epoch_end,training_status)
        return
