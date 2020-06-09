from setuptools import setup, find_packages
from distutils.util import convert_path
import sys


# ver_path = convert_path('sim_common/version.py')
# with open(ver_path) as ver_file:
#     ns = {}
#     exec(ver_file.read(), ns)
#     version = ns['version']

setup(
    name='odessa',
    version='0.1.0',
    description='ODE System Solving Accelerator - A Numba-powered modular solver for ODEs',
    author='Delft Aerospace Rocket Engineering',
    author_email='sarrazin.nathan@gmail.com',
    # url='',

    install_requires=['numpy',
                      'numba',
                      'scipy',
                      'pandas',
                      'pytest'],

    packages=find_packages('.', exclude=["test"]),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha']
)