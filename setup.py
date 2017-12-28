import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

setup(
    name="Pyspatialml",
    version="0.1",
    author="Steven Pawley",
    author_email="steven.pawley@gmail.com",
    description=("Machine learning for GIS spatial modelling"),
    license="GNU",
    keywords="grass gis",
    url="https://github.com/stevenpawley/Pyspatialml",
    packages=["Pyspatialml"]
)
