# Python predefined function setuptools
from setuptools import find_packages, setup 

setup(
    name = 'Medical Chatbot Project', 
    version= '0.0.0',
    author= 'Praveen Tiwari',
    author_email= 'praveeniiscai@gmail.com',
    packages= find_packages(),
    install_requires = []

) # setup project name,version,peackage,auther details
# find_packages() function looking constructor(__init__.py) of 
# all folder and import peackages(fun) from that constructor to in my local peackage 
# How setup.py file setup all thing??
# in our requirementa.txt has (-e .) , this always looking setup.py file and install everything
# After running the setup.py file, one file projectname.egg-info autometic generated for saving meta data of installing