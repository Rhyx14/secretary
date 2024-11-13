import torch
import einops
def fuse_conv2d(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    if isinstance(conv,torch.nn.Conv2d):
        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean)/var_sqrt * beta + gamma
        fused_conv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )
    elif isinstance(conv,torch.nn.ConvTranspose2d):
        groups=conv.groups
        w = einops.rearrange(w,'(i g) out h w -> i (g out) h w',g=groups)
        w = w * (beta / var_sqrt).reshape([1, conv.out_channels, 1, 1])
        w = einops.rearrange(w,'i (g out) h w -> (i g) out h w',g=groups)
        b = (b - mean)/var_sqrt * beta + gamma
        fused_conv = torch.nn.ConvTranspose2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=True,
            padding_mode=conv.padding_mode
        )

    fused_conv.weight = torch.nn.Parameter(w)
    fused_conv.bias = torch.nn.Parameter(b)
    return fused_conv