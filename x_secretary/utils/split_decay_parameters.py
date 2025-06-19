import torch
def split_decay_parameters(module:torch.nn.Module) -> tuple:
    '''
    splite parameters

    @return : decay, // including CONV, Linear, or module has split_decay_parameters()
            
            : no_decay, // including BN, bias of each layer, or module has split_decay_parameters()
    '''
    params_decay = []
    params_no_decay = []

    if isinstance(module, torch.nn.Linear):
        params_decay.append(module.weight)
        if module.bias is not None:
            params_no_decay.append(module.bias)

    elif isinstance(module, torch.nn.modules.conv._ConvNd):
        params_decay.append(module.weight)
        if module.bias is not None:
            params_no_decay.append(module.bias)

    elif isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,)):
        params_no_decay.extend([*module.parameters()])

    elif hasattr(module,'split_decay_parameters'):
        _decay,_no_decay=module.split_decay_parameters()
        params_decay.extend([*_decay])
        params_no_decay.extend([*_no_decay])

    else: # for default settings, no decay for immediate parameters. Then recurs its sub-module.
        for _param in module._parameters.values():
            params_no_decay.append(_param)
        for _m in module.children():
            _decay,_no_decay=split_decay_parameters(_m)
            params_decay.extend([*_decay])
            params_no_decay.extend([*_no_decay])
    
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay