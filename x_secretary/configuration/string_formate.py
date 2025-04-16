import torch
from io import StringIO
def string_formate(cfg) -> str:
    normal_properties=StringIO()
    normal_properties.write('## properties\n|name|value|\n|-|-|\n')
    
    multi_line=StringIO()

    modules=StringIO()
    modules.write('\n## torch modules\n')
    for _k,_v in cfg.__dict__.items():
        # hidden properties
        if str.startswith(_k,'_'):continue

        elif isinstance(_v,(torch.nn.Module)):
            modules.write(f'\n### {_k}\n```\n{str(_v)}\n```\n')  
        elif isinstance(_v,(torch.Tensor)):
            normal_properties.write(f"|{_k}|torch.Tensor<shape={_v.shape}, dtype={_v.dtype}>|\n")
        else:
            _str=str(_v)
            if _str.count('\n') >=1:
                multi_line.write(f'\n### {_k}\n```\n{_str}\n```\n')
            else:
                normal_properties.write("|%s|%s|\n" % (_k,_str))
    return normal_properties.getvalue() + multi_line.getvalue() + modules.getvalue()