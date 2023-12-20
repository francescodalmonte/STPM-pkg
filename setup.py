from setuptools import setup

setup(name='STPM-pkg',
      version='1.0',
      author='Francesco Dalmonte',
      packages=['STPM_model',
                'STPM_model.test'],
      scripts=['bin/train.py']
      )