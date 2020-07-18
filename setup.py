from setuptools import setup

long_description = """
This is a library (very much still in development) intended to make
sensitivity analysis easier for optimization problems, particularly
variational approximations.

For some background and motivations, see our preprint:

Covariances, Robustness, and Variational Bayes
Ryan Giordano, Tamara Broderick, Michael I. Jordan
https://arxiv.org/abs/1709.02536
"""

setup(name='LinearResponseVariationalBayes',
      version='0.2.1',
      description='Helper functions for linear response variational Bayes',
      long_description=long_description,
      url='https://github.com/rgiordan/LinearResponseVariationalBayes.py',
      author='Ryan Giordano',
      author_email='rgiordan@gmail.com',
      license='Apache 2.0',
      packages=['LinearResponseVariationalBayes'],

      python_requires='>=3',
      classifiers = [
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics'
      ],

      # This package is no longer under development, and cannot be guaranteed
      # to work with later versions of these dependent packages.
      install_requires = [
        'autograd>=1.3.0,<1.4.0',
        'numpy>=1.13.0,<=1.16.0',
        'scipy>=0.19.0,<1.0.0',
        'json_tricks<=3.11.0'
      ]
)
