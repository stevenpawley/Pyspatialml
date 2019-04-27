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
        'numpy>=1.10.0',
        'scipy>1.0.0',
        'tqdm>=4.20',
        'rasterio>=1.0',
        'pandas>=0.20',
        'shapely>=1.6',
        'geopandas>=0.3',
        'matplotlib>=2.2.4'],
    python_requires='>=3.5',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
