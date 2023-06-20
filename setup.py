from setuptools import setup
from pkg_resources import parse_requirements
with open('requirements.txt') as root:
    requirements = [str(req) for req in parse_requirements(root)]

setup(
    name='py_common',
    version='0.0.1',
    packages=['py_common'],
    url='',
    license='',
    author='CielAl',
    author_email='',
    description='',
    install_requires=requirements,
)
