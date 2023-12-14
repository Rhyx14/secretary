from typing import Any


class Transform_Wrapper_Base():
    def __init__(self,transform) -> None:
        self._transform=transform
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._transform(*args,**kwds)

class Bi(Transform_Wrapper_Base):
    pass

class Tri(Transform_Wrapper_Base):
    pass

class Quadr(Transform_Wrapper_Base):
    pass

class Quint(Transform_Wrapper_Base):
    pass


class Union_Transforms():
    def __init__(self,transform_list) -> None:
        if transform_list is None:
            self._list=[]
        else:
            self._list=transform_list
    
    @property
    def transforms(self):
        return self._list

    def __call__(self, *args: Any) -> Any:
        _args=list(args)

        for _union_transform in self._list:
            _id_start=0
            
            for _trans in _union_transform:    

                if _id_start>=len(args):
                    raise ValueError(f'Invalid args or transforms number, args: {len(args)} with transforms: {len(_trans)}.')
                
                if _trans is None:
                    _id_start +=1
                elif isinstance(_trans,Bi):
                    _args[_id_start:_id_start+2]=_trans(*_args[_id_start:_id_start+2])
                    _id_start +=2
                elif isinstance(_trans,Tri):
                    _args[_id_start:_id_start+3]=_trans(*_args[_id_start:_id_start+3])
                    _id_start +=3
                elif isinstance(_trans,Quadr):
                    _args[_id_start:_id_start+4]=_trans(*_args[_id_start:_id_start+4])
                    _id_start +=4
                elif isinstance(_trans,Quint):
                    _args[_id_start:_id_start+5]=_trans(*_args[_id_start:_id_start+5])
                    _id_start +=5
                else:
                    _args[_id_start]=_trans(_args[_id_start])
                    _id_start +=1
        
        return tuple(_args)
