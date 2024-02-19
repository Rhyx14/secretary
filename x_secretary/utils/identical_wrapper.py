from collections import defaultdict
import torch
from typing import Any
class Identical_Wrapper(torch.nn.Module):
    
    id_map=defaultdict(lambda : 0)
    def __init__(self,hooks=None,name='idt') -> None:
        '''
        等价包装器,不执行任何运算
        
        hooks: 获取输入 list[ callable(input)]
        '''
        super().__init__()

        self.comments=name
        self.id=Identical_Wrapper.id_map[name]
        Identical_Wrapper.id_map[name]+=1

        self.state_hooks=[]
        if hooks is not None:
            assert isinstance(hooks,list)
            self.state_hooks.extend(hooks)
        pass

    def __call__(self, x,*args: Any, **kwds: Any) -> Any:

        for hooks in self.state_hooks: hooks(x)
        return x
