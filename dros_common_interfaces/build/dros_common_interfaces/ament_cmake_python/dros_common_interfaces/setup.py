from setuptools import find_packages
from setuptools import setup

setup(
    name='dros_common_interfaces',
    version='0.0.0',
    packages=find_packages(
        include=('dros_common_interfaces', 'dros_common_interfaces.*')),
)
