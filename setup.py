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
        'numpy',
        'scipy',
        'tqdm',
        'rasterio>=1.0',
        'pandas',
        'shapely',
        'geopandas',
        'matplotlib',
        'scikit-learn'],
    python_requires='>=3.9',

    # testing
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
