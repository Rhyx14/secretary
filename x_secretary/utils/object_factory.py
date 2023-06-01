class object_factory():
    '''
    对象工厂，用于延迟构建对象和临时复盖构造参数
    只支持关键字参数
    '''
    def __init__(self,object_type,**kwds):
        self._object_name=object_type
        self._kwds={} | kwds
    
    def __call__(self, **kwds):
        return self._object_name(
                **(self._kwds|kwds)
            )