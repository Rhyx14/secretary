def deprecated(msg):
    def dec(f):
        print(f'{f.__name__}: {msg}')
        return f
    return dec