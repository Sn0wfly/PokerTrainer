"""
Setup script for PokerBot - GPU-Native Poker AI
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="poker-bot",
    version="0.1.0",
    author="PokerBot Team",
    author_email="contact@pokerbot.ai",
    description="GPU-accelerated poker AI using JAX and MCCFR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pokerbot/poker-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "gpu": [
            "jax[cuda]>=0.4.20",
            "jaxlib[cuda]>=0.4.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "poker-bot=poker_bot.cli:main",
            "poker-train=poker_bot.cli:train",
            "poker-play=poker_bot.cli:play",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 