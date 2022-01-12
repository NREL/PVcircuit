from setuptools import setup, find_packages

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="PVcircuit", 
        version='0.0.1',
        author="John Geisz",
        author_email="<john.geiz@NREL.gob>",
        description='Multijunction PV circuit model',
        long_description='Optoelectronic model of tandem and multijunction solar cells',
        packages=find_packages(),
        install_requires=['numpy>=1.13.3', 'matplotlib>=2.1.0', 'scipy>=1.0.0'],
)