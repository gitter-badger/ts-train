import pip


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line.split("\s")[0] for line in lineiter if line and not line.startswith("#")]


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

links = []
requires = parse_requirements('requirements.txt')

    
setup(
    name='ts_train',
    version="0.2",
    url='https://minesh1291.github.io/ts-train',
    license='MIT License',
    author="Minesh A. Jethva",
    author_email="minesh.1291@gmail.com",
    description='Time-Series Handling for Machine Learning Tasks',
    long_description=long_description,
    include_package_data=True,
    zip_safe=False,
    packages=["ts_train"],
    package_dir={'ts_train':'ts_train'},
    install_requires=requires,
    dependency_links=links
)

#https://packaging.python.org/tutorials/packaging-projects/#classifiers
