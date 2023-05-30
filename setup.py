import os
from setuptools import setup, find_packages


version = 0.01
with open(os.path.join('sparsemeta', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

setup(
    name="sparsemeta",
    version=version,
    packages=find_packages(),
)
