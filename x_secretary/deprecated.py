def deprecated(msg):
    def dec(f):
        def ff(*args,**kwargs):
            print(f'\033[0;30;43m{f.__name__}: \033[0m{msg}')
            return f(*args,**kwargs)
        return ff
    return dec