import time
def get_str_time(tpl:str="%Y-%m-%d_%H:%M:%S_%z")-> str:
    return time.strftime(tpl,time.gmtime())

from contextlib import contextmanager
import time
@contextmanager
def measure_time(logger=None,banner="Running time"):
    '''
    measure the time spans
    '''
    start_time=time.time()
    times=[]
    yield
    times.append(time.time()-start_time)

    if logger is not None:
        logger.info(f'{banner} {sum(times)/len(times)}s')
    else:
        print(f'{banner} {sum(times)/len(times)}s')

        
def measure_func_time(func:callable,repeat=1,mode='avg',logger=None,banner="running time"):
    '''
    measure the time spans

    repeat: repeat time

    mode: 'avg' average time
          'sum' total time
    '''
    times=[]
    for _ in range(repeat):
        _start_time=time.time()
        func()
        times.append(time.time()-_start_time)
    match mode:
        case 'avg':
            if logger is not None:
                logger.info(f'Average {banner} {sum(times)/len(times)}s')
            else:
                print(f'Average {banner} {sum(times)/len(times)}s')
        case 'sum':
            if logger is not None:
                logger.info(f'Total {banner} {sum(times)}s')
            else:
                print(f'Total {banner} {sum(times)}s')
        case _:
            raise ValueError(f'Erroneous mode: "{mode}"')