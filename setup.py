from setuptools import setup, find_packages
from os import path

base = path.abspath(path.dirname(__file__))

__version__ = '0.0.0'

with open(path.join(base, 'requirements.txt'), encoding='utf-8') as file:
    reqs = file.read().split('\n')

install_requires = [x.strip() for x in reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in reqs if x.startswith('git+')]

setup(
    name='ScratchML',
    version=__version__,
    description='Implementation of some of the Machine Learning models in python.',
    url='https://github.com/Imartinezcuevas/ScratchML',
    download_url='https://github.com/Imartinezcuevas/ScratchML/tarball/master',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Iván Martínez Cuevas',
    install_requires=install_requires,
    etup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
    author_email='imartinezcuevas@gmail.com'
)