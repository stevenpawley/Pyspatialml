from setuptools import setup

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
        'rasterio'],
    python_requires='>3.3'
)
