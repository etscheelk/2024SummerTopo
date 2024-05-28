from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy

chpl_libraries=["chpl", "gmp", "hwloc", "qthread", "jemalloc", "re2", "m", "pthread"]
setup(name = 'chplpitest library',
	ext_modules = cythonize(
		Extension("chplpitest",
			include_dirs=[numpy.get_include()],
			sources=["chplpitest.pyx"],
			libraries=["chplpitest"] + chpl_libraries + ["chplpitest"])))
