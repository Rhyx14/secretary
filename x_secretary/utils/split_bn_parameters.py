import torch
def split_decay_parameters(module):
    '''
    get params which will subject to param_decay 

    @return : decay, // including CONV, Linear,
            : no_decay, // including BN, bias of each layer
    '''

    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        else:
            _current_params=get_current_parameters(m)
            params_no_decay.extend(_current_params)
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

def get_current_parameters(m:torch.nn.Module):
    rslt=[]
    for k,v in m._parameters.items():
        # if isinstance(v,torch.nn.Parameter):
            rslt.append(v)
    return rslt