def solo_method(func):
    def f(self,*args,**kwargs):
        if(self.LOCAL_RANK==0):
            return func(self,*args,**kwargs)
        else:
            return None
    return f