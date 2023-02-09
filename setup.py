from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='u-umetrics',
    version='1.0.7',
    packages=['umetrics'],
    url='https://github.com/geasyheart/u-metrics',
    license='MIT',
    author='yuzhang',
    author_email='geasyheart@163.com',
    description='calculate precision or recall or f1 score on large-scale datasets',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
