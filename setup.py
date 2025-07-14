"""
Setup script for PokerBot - GPU-Native Poker AI
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Define la extensión de Cython
extensions = [
    Extension(
        "poker_bot.core.hasher",
        ["poker_bot/core/hasher.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="poker-trainer",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'click',
        'PyYAML',
        'tqdm',
        'psutil',
        'phevaluator',
        'Cython',
    ],
    ext_modules=cythonize(extensions, language_level="3"),
    entry_points={
        'console_scripts': [
            'poker-bot=poker_bot.cli:cli',
        ],
    },
    python_requires=">=3.8",
) 