# 🚀 SETUP_CYTHON.PY - Compile Cython module for maximum performance
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="poker_bot_fast_hasher",
    ext_modules=cythonize("poker_bot/fast_hasher.pyx"),
    include_dirs=[numpy.get_include()],  # Necessary for Cython to understand NumPy
    zip_safe=False,
) 