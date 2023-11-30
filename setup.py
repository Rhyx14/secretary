from setuptools import setup, find_packages
from os import path
this_directory = path.abspath(path.dirname(__file__))
long_description = None
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()
 
setup(
      name='x_secretary', # 包名称
      packages=find_packages(exclude=['__pycache__']), # 需要处理的包目录
      version='0.2.0', # 版本
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python', 'Intended Audience :: Developers',
        #   'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.10',
      ],
      install_requires=['torch'],
      extras_requires=['opencv-python','accelerate'],
      scripts=['bin/xsrun'],
    #   entry_points={'console_scripts': ['pmm=pimm.pimm_module:main']},
    #   package_data={'': ['*.json']},
      author='rhyx14', # 作者
      description='Auxiliary tools for pytorch training.', # 介绍
    #   long_description=long_description, # 长介绍，在pypi项目页显示
    #   long_description_content_type='text/markdown', # 长介绍使用的类型
      url='https://github.com/Rhyx14/secretary', # 包主页，一般是github项目主页
      license='MIT', # 协议
      python_requires='>=3.10'
    #   keywords='pimm source manager'  # 关键字 搜索用
    )