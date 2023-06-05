from setuptools import setup,find_packages

setup(
    name='marugoto_mods',
    version=__version__,

    url='https://github.com/brooksja/marugoto_mods.git',
    author='James Brooks',
    author_email='jamesalexander.brooks@med.uni-duesseldorf.de',

    py_modules=find_packages(),
  
    install_requires=[
      'https://github.com/KatherLab/marugoto.git',
    ],
)
