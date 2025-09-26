import torch,einops
from functools import partial

def get_scale_factor(q:float,original_weight:torch.Tensor,lower_bound,upper_bound):

    top9=torch.abs(original_weight).quantile(dim=1,q=q)
    scale_factor= upper_bound / top9
    
    new_weight = torch.clamp(torch.round(scale_factor[:,None] * original_weight),lower_bound,upper_bound)
    reconstruct = new_weight / scale_factor[:,None]
    score= torch.nn.functional.mse_loss(original_weight,reconstruct)

    return score, scale_factor

import numpy as np
def search_mean(search_range:list,step,eval:callable,**kwds):
    min_score=None
    min_results=None
    for _i in np.arange(search_range[0],search_range[1],step):
        _score,_results=eval(_i,**kwds)
        if min_score is None or _score< min_score:
            min_score=_score
            min_results=_results
    return min_results

@torch.no_grad
def symmetric_quantize_weight(conv_group,lower_bound,upper_bound,search_range=(0.97,1.0),step=0.001) -> tuple[float, torch.Tensor]:
    '''
    Symmetric quantize weight of an conv layer, channel-wize

    @return : (scale_factor, new_weight)
    '''
        
    original_list=[]
    for _conv in conv_group:
        if isinstance(_conv,torch.nn.ConvTranspose2d):
            # _tmp=einops.rearrange(_tmp,'in out h w ->  out (in h w)')
            original_list.append(einops.rearrange(_conv.weight.data,'(in g) out h w -> (g out) (in h w)',g=_conv.groups))
        elif isinstance(_conv,torch.nn.Conv2d):
            original_list.append(einops.rearrange(_conv.weight.data,'out in h w -> out (in h w)'))
    original_weight=torch.concat(original_list,dim=1)

    return search_mean(search_range=search_range,
                step=step,
                eval=get_scale_factor,
                original_weight=original_weight,
                lower_bound=lower_bound,upper_bound=upper_bound)