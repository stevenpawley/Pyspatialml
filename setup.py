import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

setup(
    name="pyspatialml",
    version="0.1",
    author="Steven Pawley",
    author_email="steven.pawley@gmail.com",
    description=("Machine learning for GIS spatial modelling"),
    license="GNU",
    keywords="GIS",
    url="https://github.com/stevenpawley/pyspatialml",
    packages=["pyspatialml"],
    install_requires=[
        'tqdm',
        'rasterio',
        'numpy'],
    python_requires='>3.3'
)
