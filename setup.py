import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    # package metadata
    name="pyspatialml",
    version="0.21",
    author="Steven Pawley",
    author_email="steven.pawley@gmail.com",
    description=("Machine learning for GIS spatial modelling"),
    long_description=README,
    long_description_content_type="text/markdown",
    license="GNU",
    keywords="GIS",
    url="https://github.com/stevenpawley/pyspatialml",

    # files/directories to be installed with package
    packages=find_packages(),
    package_data={
        '': ['*.tif', '*.dbf', '*.prj', '*.shp', '*.shx'],
    },
    include_package_data=True,

    # package dependencies
    install_requires=[
        'numpy>=1.10.0',
        'scipy>1.0.0',
        'tqdm>=4.20',
        'rasterio>=1.0',
        'pandas>=0.20',
        'shapely>=1.6',
        'geopandas>=0.3',
        'matplotlib>=2.2.4',
        'scikit-learn>=0.22'],
    python_requires='>=3.7',

    # testing
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
