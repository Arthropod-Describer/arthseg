import numpy
import os
from glob import glob
from pathlib import Path
from setuptools import Extension, setup

# platform specific settings
if os.name == 'nt':
    flags = ['/std:c++17']
else:
    flags = ['-std=c++17']

module = Extension(
    'arthseg',
    sources=glob('arthseg/**/*.cpp', recursive=True),
    include_dirs=[numpy.get_include(), 'arthseg', 'arthseg/lib'],
    extra_compile_args=flags,
)

setup(
    name='arthseg',
    version='0.0.6',
    license='MIT',
    description='Native library for arthropod segmentation',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type='text/markdown',
    author='Matej Pekar',
    author_email='matej.pekar120@gmail.com',
    url='https://github.com/Arthropod-Describer/arthseg',
    install_requires=['numpy'],
    ext_modules=[module],
    package_data={'arthseg': ['__init__.pyi']},
    packages=['arthseg'],
)
