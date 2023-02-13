import sys
import os
ls=sys.path
path=list(filter(lambda path : path.find('site-packages')!=-1,ls))[0]
print(f'site-package folder: {path}')
current_path=os.path.abspath('.')
print(f'current path: {current_path}')

def install():
    with open(os.path.join(path,'secretary.pth'),'w') as f:
        f.writelines([
            os.path.join(current_path,'src/')
        ])
        f.flush()
    print('done')

def uninstall():
    file_path=os.path.join(path,'secretary.pth')
    if(os.path.exists(file_path)):
        os.remove(file_path)
        print(f'remove file: {file_path}')
        print('done')
    else:
        print(f'no such file:\n{file_path}')

s=input('install (i) / uninstall (r)')
if(s=='i'):
    install()
elif(s=='r'):
    uninstall()
else:
    print('invalid operation')