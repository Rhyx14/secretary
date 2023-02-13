import sys
import os
ls=sys.path
path=list(filter(lambda path : path.find('site-packages')!=-1,ls))[0]
print(f'site-package folder: {path}')
current_path=os.path.abspath('.')
print(f'current path: {current_path}')

def install():
    os.symlink(
        os.path.join(current_path,'src'),
        os.path.join(path,'secretary'),
        target_is_directory=True
    )
    print('done')

def uninstall():
    folder_path=os.path.join(path,'secretary')
    if(os.path.exists(folder_path)):
        os.remove(folder_path)
        print(f'remove symbol: {folder_path}')
        print('done')
    else:
        print(f'no such file:\n{folder_path}')

s=input('install (i) / uninstall (r)')
if(s=='i'):
    install()
elif(s=='r'):
    uninstall()
else:
    print('invalid operation')