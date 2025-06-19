import torch
from functools import partial
def get_results(q,weight,lower_bound,upper_bound,is_transposed=False):
    if is_transposed:
        top9=torch.abs(weight).transpose(0,1).flatten(1).quantile(dim=1,q=q)
        scale_factor= upper_bound / top9
        new_weight= torch.clamp(torch.round(scale_factor[None,:,None,None] * weight),lower_bound,upper_bound)
        reconstruct = new_weight / scale_factor[None,:,None,None]
    else:
        top9=torch.abs(weight).flatten(1).quantile(dim=1,q=q)
        scale_factor= upper_bound / top9
        new_weight= torch.clamp(torch.round(scale_factor[:,None,None,None] * weight),lower_bound,upper_bound)
        reconstruct = new_weight / scale_factor[:,None,None,None]
    score= torch.nn.functional.mse_loss(weight,reconstruct)
    return score, (scale_factor,new_weight)

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

def symmetric_quantize_weight(conv,lower_bound,upper_bound,search_range=(0.97,1.0),step=0.001) -> tuple[float, torch.Tensor]:
    '''
    Symmetric quantize weight of an conv layer, channel-wize

    @return : (scale_factor, new_weight)
    '''
    return search_mean(search_range=search_range,
                step=step,
                eval=partial(get_results,is_transposed=isinstance(conv,torch.nn.ConvTranspose2d)),
                weight=conv.weight.data,lower_bound=lower_bound,upper_bound=upper_bound)