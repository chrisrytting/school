from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
from os import system


shared_obj = "gcc cssor.c -fPIC -c -o cssor.o"
print shared_obj
system(shared_obj)

ext_modules=[Extension(
						"cython_cssor",
						["cython_cssor.pyx"],

						extra_link_args=["cssor.o"])]

setup(name="cython_cssor",
	cmdclass={'build_ext':build_ext},
	include_dirs=[get_include()],
	ext_modules=ext_modules)
