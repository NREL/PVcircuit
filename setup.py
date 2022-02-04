from setuptools import setup, find_packages

# Setting up
setup(
        name="PVcircuit", 
        version='0.0.2',
        author="John Geisz",
        author_email="<john.geiz@NREL.gov>",
        description='Multijunction PV circuit model',
        long_description='Optoelectronic model of tandem and multijunction solar cells',
        url='https://github.nrel.gov/Tandems/PVcircuit',
        license='LICENSE.txt',
        packages=find_packages(),
        install_requires=['numpy>=1.13.3', 
                          'matplotlib>=2.1.0', 
                          'scipy>=1.0.0',
                          'parse>=1.19.0',
                          'ipywidgets>=7.6.5',
                          'pandas>=1.0'],
)
