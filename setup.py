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
      version='0.2.0',
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

      install_requires = [
        'autograd',
        'numpy',
        'scipy'
      ]
)
