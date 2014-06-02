from distutils.core import setup
from Cython.Build import cythonize
from numpy import get_include

setup(
    ext_modules = cythonize('diversity.pyx'),
    include_dirs = [get_include()]
)
