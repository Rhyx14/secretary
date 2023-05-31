import time
def get_str_time(tpl:str="%Y-%m-%d_%H:%M:%S_%z")-> str:
    return time.strftime(tpl,time.gmtime())