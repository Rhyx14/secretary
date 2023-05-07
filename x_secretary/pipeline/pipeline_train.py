import logging
import torch
from torch.utils.data.dataloader import DataLoader
from .pipeline_base import PipelineBase
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast

class Image_training(PipelineBase):
    def __init__(self,
        logger,
        cfg,
        before_hooks=None,
        after_hooks=None,
        before_epoch_hooks=None,
        after_epoch_hooks=None,
        before_turn_hooks=None,
        after_turn_hooks=None,
        mode:str='classification'
        ):
        '''
        training pipeline for pytorch distribution mode,

        cfg should contains following keys:
        {
            'train_dataset':torch.data.Dataset,
            'net':torch.nn.Module,
            'loss':torch.nn.Module,
            'opt':torch.optim.Optimizer,
            'BATCH_SIZE':int,
            'HALF':bool,
            'EPOCH':int
        }

        ---------
        before_hooks : hooks before training, with parameter (self)

        after_hooks : hooks after training, with parameter (self)

        before_epoch_hooks : hooks before each epoch, with parameter (configuration)

        after_epoch_hooks : hooks after each epoch, with parameter (configuration,loss,ep)

        before_turn_hooks : hooks before each training turn, with parameter (configuration)

        after_turn_hooks : hooks after each training turn, with parameter (configuration,batch_id,loss,ep)

        mode: 'classification' or 'segmentation'
        '''
        self.cfg=cfg
        self.logger=logger
        self.before_epoch_hooks=before_epoch_hooks
        self.after_epoch_hooks=after_epoch_hooks
        self.before_turn_hooks=before_turn_hooks
        self.after_turn_hooks=after_turn_hooks

        self._Check_Attribute(self.cfg,'net',(torch.nn.DataParallel,torch.nn.Module))
        self._Check_Attribute(self.cfg,'loss',(torch.nn.Module,))
        self._Check_Attribute(self.cfg,'opt',(torch.optim.Optimizer,))
        self._Check_Attribute(self.cfg,'train_dataset',(torch.utils.data.Dataset,))
        self._Check_Attribute(self.cfg,'BATCH_SIZE',(int,))
        self._Check_Attribute(self.cfg,'EPOCH',(int,))
        self._Check_Attribute(self.cfg,'HALF',(bool,))

        if(mode=='classification'):
            self.Run=self._Run_cls
        elif mode=='segmentation':
            self.Run=self._Run_seg
        else:
            raise NotImplementedError(f'Pipleline for {mode} hasn''t been implemented yet.')
        
        super().__init__(self.logger,cfg.net,before_hooks,after_hooks)
    
        
    def _Run_seg(self,dl_workers=2,*args,**kwargs):

        cfg=self.cfg

        dl=DataLoader(cfg.train_dataset,batch_size=cfg.BATCH_SIZE,num_workers=dl_workers,prefetch_factor=4,shuffle=True,drop_last=True)
        cfg.net.train()
        scaler=GradScaler()

        for ep in range(cfg.EPOCH):
            # dl.sampler.set_epoch(ep)
            for _b_id,datum in enumerate(dl):

                x=datum['X'].cuda()
                y=datum['Y'].cuda()

                PipelineBase.call_hooks(self.before_turn_hooks,self.cfg)

                cfg.opt.zero_grad()

                if(cfg.HALF):
                    with autocast():
                        _out=cfg.net(x)
                        _loss = cfg.loss(_out,y)
                    scaler.scale(_loss).backward()
                    # torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
                    scaler.step(cfg.opt)
                    scaler.update()
                else:
                    _out=cfg.net(x)
                    _loss = cfg.loss(_out,y)
                    _loss.backward()
                    cfg.opt.step()

                PipelineBase.call_hooks(self.after_turn_hooks,self.cfg,_b_id,_loss.item(),ep)

            if hasattr(cfg,'lr_scheduler'):
                cfg.lr_scheduler.step()

            PipelineBase.call_hooks(self.after_epoch_hooks,self.cfg,_loss,ep)

        return


    def _Run_cls(self,dl_workers=2,*args,**kwargs):
        cfg=self.cfg

        dl=DataLoader(cfg.train_dataset,batch_size=cfg.BATCH_SIZE,num_workers=dl_workers,prefetch_factor=4,shuffle=True,drop_last=True)
        cfg.net.train()
        scaler=GradScaler()

        for ep in range(cfg.EPOCH):
            # dl.sampler.set_epoch(ep)
            for _b_id,(x,y) in enumerate(dl):

                x=x.cuda(non_blocking=True)
                y=y.cuda(non_blocking=True)
                PipelineBase.call_hooks(self.before_turn_hooks,self.cfg)

                cfg.opt.zero_grad()

                if(cfg.HALF):
                    with autocast():
                        _out=cfg.net(x)
                        _loss = cfg.loss(_out,y)
                    scaler.scale(_loss).backward()
                    # torch.nn.utils.clip_grad_value_(net.parameters(), 1.0)
                    scaler.step(cfg.opt)
                    scaler.update()
                else:
                    _out=cfg.net(x)
                    _loss = cfg.loss(_out,y)
                    _loss.backward()
                    cfg.opt.step()

                PipelineBase.call_hooks(self.after_turn_hooks,self.cfg,_b_id,_loss.item(),ep)

            if hasattr(cfg,'lr_scheduler'):
                cfg.lr_scheduler.step()

            PipelineBase.call_hooks(self.after_epoch_hooks,self.cfg,_loss,ep)

        return
