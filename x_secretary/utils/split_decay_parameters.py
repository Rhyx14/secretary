import torch
def split_decay_parameters(module:torch.nn.Module) -> tuple:
    '''
    splite parameters

    @return : decay, // including CONV, Linear,
            : no_decay, // including BN, bias of each layer
    '''
    params_decay = []
    params_no_decay = []

    if isinstance(_m, torch.nn.Linear):
        params_decay.append(_m.weight)
        if _m.bias is not None:
                params_no_decay.append(_m.bias)

    elif isinstance(_m, torch.nn.modules.conv._ConvNd):
        params_decay.append(_m.weight)
        if _m.bias is not None:
            params_no_decay.append(_m.bias)

    elif isinstance(_m, (torch.nn.modules.batchnorm._BatchNorm,)):
        params_no_decay.extend([*_m.parameters()])

    elif hasattr(module,'split_decay_parameters'):
        _decay,_no_decay=module.split_decay_parameters()
        params_decay.extend([*_decay])
        params_no_decay.extend([*_no_decay])

    else: # for default settings, no decay for immediate parameters. Then recurs its sub-module.
        for _2_param in _m._parameters.values():
            params_no_decay.append(_2_param)
        for _m in module.children():
            _decay,_no_decay=split_parameters_for_decay(_m)
            params_decay.extend([*_decay])
            params_no_decay.extend([*_no_decay])
    
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay