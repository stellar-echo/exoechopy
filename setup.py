#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#                                                                              #
#                       OOOOOOOOO---------------OOOOOOOOO                      #
#                       OOOOO-----------------------OOOOO                      #
#                       OOO-------------------NNNN----OOO                      #
#                       O--------------------NNNNNN-----O                      #
#                       O----------NN--------NNN-NNN----O                      #
#                       ---------NNNNNNN-----NNN--NNN----                      #
#                       ---------NNNNNNNNN---NNN---NNNNNN                      #
#                       NNNNNN---NNN---NNNNNNNNN---------                      #
#                       ----NNN--NNN-----NNNNNN----------                      #
#                       O----NNN-NNN--------NN----------O                      #
#                       O-----NNNNNN--------------------O                      #
#                       OOO----NNNN-------------------OOO                      #
#                       OOOOO-----------------------OOOOO                      #
#                       OOOOOOOOO---------------OOOOOOOOO                      #
#                                                                              #
#                                Nanohmics, Inc.                               #
#                                                                              #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


from setuptools import setup, find_packages


# configure setup
setup(
    name='exoechopy',
    version='0.1',
    description='Toolkit for simulating and analyzing stellar variability and resulting echoes',
    long_description='',
    author='Nanohmics, Inc.',
    author_email='cmann@nanohmics.com',
    packages=find_packages(),
    test_suite='tests.test_suite',
    dependency_links=[],
    python_requires='>=3',
    install_requires=[
    'numpy >= 1.11',
    'scipy >= 0.18',
    'matplotlib >= 2.0',
    'astropy >= 3.0.4'],
    extras_require={
    'docs': [
      'sphinx >= 1.5',
      'sphinx_rtd_theme >= 0.1.9', ]},
    keywords=['astronomy', 'astrophysics', 'cosmology', 'space', 'science',
              'units', 'table', 'wcs', 'samp', 'coordinate', 'fits',
              'modeling', 'models', 'fitting', 'exoplanet', 'detection'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    include_package_data=True,
    zip_safe=False)

