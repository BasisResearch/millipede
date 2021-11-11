import re
import sys

from setuptools import find_packages, setup

with open("millipede/__init__.py") as f:
    for line in f:
        match = re.match('^__version__ = "(.*)"$', line)
        if match:
            __version__ = match.group(1)
            break

try:
    long_description = open("README.md", encoding="utf-8").read()
except Exception as e:
    sys.stderr.write("Failed to read README.md: {}\n".format(e))
    sys.stderr.flush()
    long_description = ""

setup(
    name="millipede",
    version=__version__,
    description="A library for bayesian variable selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["millipede"]),
    url="https://github.com/broadinstitute/millipede",
    author="Martin Jankowiak",
    author_email="mjankowi@broadinstitute.org",
    install_requires=[
        "torch>=1.9",
        "pandas",
        "polyagamma==1.3.2",
        "tqdm",
    ],
    extras_require={
        "test": [
            "isort>=5.0",
            "flake8",
            "pytest>=5.0",
        ],
    },
    python_requires=">=3.8",
    keywords="bayesian variable selection",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.8",
    ],
)
