import os,sys,platform,torch,re,subprocess
LINE='============================================================================='
def get_cpu_name()->str:
    rslt=''
    try:
        if sys.platform == 'linux':
            _strs=subprocess.check_output(['cat','/proc/cpuinfo'])
            _strs=bytes.decode(_strs)
            rslt=re.search(r"model name\t: ([\w\s\(\)]*)\n",_strs,re.S).group(1)
            # _strs=os.system('cat/cpuinfo | grep "model name"')
            pass
    except:
        pass
    finally:
        return rslt


def get_sys_info() -> str:
    '''
    获取系统环境信息（目前只支持CUDA环境）
    '''
    cuda_devices=[
        str(torch.cuda.get_device_properties(i))
        for i in range(torch.cuda.device_count())
    ]
    s=[
        f'Operation System: {platform.platform()}  {platform.version()}',
        f'Host Name: {platform.node()}',
        LINE,
        f'CPU: {get_cpu_name()}',
        f'CUDA devices:',
        *cuda_devices,
        LINE,
        f'Python: {sys.version}',
        f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}, HIP version: {torch.version.hip}',
        f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","undefined")}',
        f'NVIDIA_TF32_OVERRIDE={os.environ.get("NVIDIA_TF32_OVERRIDE","undefined")}',
    ]
    return '\n'.join(s)

def get_host_name() ->str:
    '''
    get host name, (computer name in Windows)
    '''
    return platform.node()
