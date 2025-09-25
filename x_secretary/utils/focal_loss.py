import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
# FocalLoss, from:
# https://zhuanlan.zhihu.com/p/73965733
class FocalLoss(nn.Module):

    def __init__(self,
                 mode='CE', n_classes=21, mean=True,
                 gamma=2, eps=1e-7):
        '''
        FocalLoss, from:
        
        https://zhuanlan.zhihu.com/p/73965733

        When mode is set to 'CE', category index 255 will be ignored as so does in Cityscapes dataset.
        '''
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mode = mode
        self.n_classes = n_classes
        self.mean = mean

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        """

        :param input: [bs, c, h, w],
        :param target: [bs, c, h, w]
        :return: tensor
        """

        if self.mode == 'BCE':
            # target = label_to_one_hot(target, n_class=self.n_class)
            # pt = input.sigmoid()

            BCE = nn.BCEWithLogitsLoss(reduction='none')(input, target)
            loss = torch.abs(target - F.softmax(input,dim=1)) ** self.gamma * BCE

        elif self.mode == 'CE':
            if input.dim() > 2:
                input= einops.rearrange(input,'b c h w -> (b h w) c')
                target = einops.rearrange(target,'b h w -> (b h w)')
                # input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
                # target = target.reshape(-1, 1)
            # print(input.shape, target.shape)
                
            _input = input[target!=255,:] # [? c]
            _target= target[target!=255] # [?]
            # _target= F.one_hot(_target,num_classes=self.n_classes) # [? c]
            
            pt = _input.softmax(dim=1)
            pt = pt.gather(dim=1, index=_target.unsqueeze(-1)).view(-1) # [?]

            # print(f'pt:{pt.shape}')
            CE = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(_input, _target) #[?]
            # print(f'CE:{CE.shape}')
            # print(CE.shape, pt.shape)
            loss = (1 - pt) ** self.gamma * CE
        else:
            raise Exception(f'*** focal loss mode:{self.mode} wrong!')

        if self.mean:
            return loss.mean()
        else:
            return loss.sum()