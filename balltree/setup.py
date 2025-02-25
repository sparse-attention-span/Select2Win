from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extra_compile_args = ['-fopenmp', '-O3']
extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "balltree",
        ["balltree.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++"
    )
]

setup(
    name="balltree",
    ext_modules=cythonize(ext_modules),
    install_requires=['numpy']
)