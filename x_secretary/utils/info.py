import os,sys,platform,torch
def get_sys_info() -> str:
    '''
    获取系统环境信息（只支持CUDA环境）
    '''
    cuda_devices=[
        str(torch.cuda.get_device_properties(i))
        for i in range(torch.cuda.device_count())
    ]
    s=[
        '==================================================================================',
        f'Operation System: {platform.platform()}  {platform.version()}',
        f'Host Name: {platform.node()}',
        f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}, HIP version: {torch.version.hip}',
        f'Python: {sys.version}',
        f'CUDA devices:',
        *cuda_devices,
        f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","undefined")}',
        '==================================================================================\n'
    ]
    return '\n'.join(s)

def get_host_name() ->str:
    '''
    get host name, (computer name in Windows)
    '''
    return platform.node()
