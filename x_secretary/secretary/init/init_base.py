class init_base():
    def __call__(self,secretary_object):
        secretary_object.__dict__.update(self._updated)
        pass