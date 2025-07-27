from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))

# Read the long description from README.md
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='x_secretary',  # Package name
    packages=find_packages(exclude=['__pycache__']),  # Include all packages except __pycache__
    version='0.3.250727',  # Version
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'torch>=2.0.1',
        'opencv-python',
        'tqdm',
        'pyyaml',
        'accelerate>=1.1.0',
        'torchvision>=0.15',
        'pytorch_warmup',
        'loguru',
    ],
    scripts=['bin/xsrun', 'bin/xs-draw'],  # Executable scripts
    author='rhyx14',  # Author
    description='Auxiliary tools for pytorch training.',  # Short description
    long_description=long_description,  # Long description from README.md
    long_description_content_type='text/markdown',  # Format of the long description
    url='https://github.com/Rhyx14/secretary',  # Project homepage
    license='MIT',  # License
    python_requires='>=3.10',  # Python version requirement
    keywords='pytorch training tools machine-learning deep-learning',  # Keywords
    # package_data={'': ['*.json']},  # Include non-Python files if needed
    # entry_points={'console_scripts': ['x_secretary=x_secretary.cli:main']},  # Console scripts
)