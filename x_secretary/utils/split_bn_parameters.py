import torch
from torchvision import models
def split_bn_parameters(module):
    '''
    get params of SNN

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
        elif isinstance(m, (torch.nn.modules.batchnorm._BatchNorm)):
            params_no_decay.extend([*m.parameters()])

        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay