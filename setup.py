import numpy
from glob import glob
from setuptools import Extension, setup
from pathlib import Path


module = Extension(
    'arthseg',
    sources=glob('arthseg/**/*.cpp', recursive=True),
    include_dirs=[numpy.get_include(), 'arthseg', 'arthseg/lib'],
    extra_compile_args=['-std=c++20'],
)

setup(
    name='arthseg',
    version='0.0.1',
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
