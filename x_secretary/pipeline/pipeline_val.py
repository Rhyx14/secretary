import torch
from .pipeline_base import PipelineBase
from torch.utils.data.dataloader import DataLoader
from ..utils.segment_metric import Metric

class ImageSegmentation_Val_Pipeline(PipelineBase):
    '''
    val pipline for image classification

    before_turn_hooks : hooks before each training turn, with parameter ()

    after_turn_hooks : hooks after each training turn, with parameter (batch_id)
    '''
    def __init__(self,
        logger,
        batch_size,
        n_classes,
        net:torch.nn.Module,
        dataset,
        cuda_device:list=None,
        dl_workers=2,
        dl_prefetch_factor=10,
        before_hooks=None,
        after_hooks=None,
        before_turn_hooks=None,
        after_turn_hooks=None,):
        super().__init__(logger,net,before_hooks,after_hooks)
        
        self.batch_size=batch_size
        self.cuda_device=cuda_device
        self.n_classes=n_classes
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor

        self.before_turn_hooks=before_turn_hooks
        self.after_turn_hooks=after_turn_hooks

    def Run(self,epoch=-1,*args,**kwargs):

        metric=Metric(self.n_classes)
        dl=DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.dl_workers,prefetch_factor=self.dl_prefetch_factor,pin_memory=False)
        with torch.no_grad():
            for iter, datum in enumerate(dl):
               inputs = datum['X'].cuda()
               gt=datum['l'].cpu().numpy()

               # simulate snn
               PipelineBase.call_hooks(self.before_turn_hooks)
               output = self.net(inputs)
               output = output.data.cpu().numpy()

               pred=output.argmax(axis=1)
               
               metric.add_batch(gt,pred)
               PipelineBase.call_hooks(self.after_turn_hooks,iter)
               # N, _, h, w = output.shape
               # pred = output.transpose(0, 2, 3, 1).reshape(-1, N_CLASS).argmax(axis=1).reshape(N, h, w)

               # target = datum['l'].cpu().numpy().reshape(N, h, w)
               # for p, t in zip(pred, target):
               #     total_ious.append(evaluator.iou(p, t))
               #     pixel_accs.append(evaluator.pixel_acc(p, t))

            # Calculate average IoU
            # total_ious = np.array(total_ious).T  # n_class * val_len
            # ious = np.nanmean(total_ious, axis=1)
            # pixel_accs = np.array(pixel_accs).mean()
            # logger.info("epoch{}, pix_acc: {}, meanIoU: {}".format(epoch, pixel_accs, np.nanmean(ious)))
            self.logger.info("epoch: {}, pix_acc: {}, meanIoU: {}".format(epoch, metric.Pixel_Accuracy(),metric.Mean_Intersection_over_Union()))

class Image_Val_Pipeline(PipelineBase):
    '''
    val pipline for image classification

    before_turn_hooks : hooks before each training turn, with parameter ()

    after_turn_hooks : hooks after each training turn, with parameter (batch_id,loss)
    '''
    def __init__(self,
        logger,
        batch_size,
        net:torch.nn.Module,
        dataset,
        cuda_device:list=None,
        dl_workers=2,
        dl_prefetch_factor=4,
        before_hooks=None,
        after_hooks=None,
        before_turn_hooks=None,
        after_turn_hooks=None,
        ):
        super().__init__(logger,net,before_hooks,after_hooks)

        self.batch_size=batch_size
        self.cuda_device=cuda_device
        self.dataset=dataset
        self.dl_workers=dl_workers
        self.dl_prefetch_factor=dl_prefetch_factor
        self.before_turn_hooks=before_turn_hooks
        self.after_turn_hooks=after_turn_hooks

    def Run(self,loss=None,*args,**kwargs):
        dl=DataLoader(self.dataset,batch_size=self.batch_size,num_workers=self.dl_workers,prefetch_factor=self.dl_prefetch_factor,pin_memory=True)
        acc=0.0
        _loss=0.0
        with torch.no_grad():
            for _bid,(x,label) in enumerate(dl):
                if(self.cuda_device is not None):
                    x=x.cuda(self.cuda_device[0])
                    label=label.cuda(self.cuda_device[0])

                # simulate snn
                PipelineBase.call_hooks(self.before_turn_hooks)
                _out=self.net(x)

                if loss is not None:
                    _loss = (_loss*_bid + loss(_out,label).item())/(_bid+1)

                pred = _out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                acc += pred.eq(label.view_as(pred)).sum().item() 
                
                PipelineBase.call_hooks(self.after_turn_hooks,_bid,_loss)
                # pred=torch.argmax(_out.squeeze(),dim=-1)
                # tmp=torch.where(label==pred,1,0)
                # acc+=torch.sum(tmp).item()

        r=acc/float(len(self.dataset))
        return acc, r, _loss