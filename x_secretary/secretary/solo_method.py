def solo_method(func):
    def f(self,*args,**kwargs):
        if(self.LOCAL_RANK==0):
            return func(self,*args,**kwargs)
        else:
            return None
    return f

def solo_method_with_default_return(ret_value):
    def _solo(func):
        def f(self,*args,**kwargs):
            if(self.LOCAL_RANK==0):
                return func(self,*args,**kwargs)
            else:
                return ret_value
        return f
    return _solo

def solo_chaining_method(func):
    def f(self,*args,**kwargs):
        if(self.LOCAL_RANK==0):
            return func(self,*args,**kwargs)
        else:
            return self
    return f