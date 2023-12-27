from setuptools import setup, find_packages
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
    
__version__ = "0.0.0"

REPO_NAME = "Kidney-Disease-Classification"
AUTHOR_NAME = "ShubhamPawar-3333"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "shubham.d.pawar.3333@gmail.com"


# function to get list of packages from file
def get_requirements(filepath:str) -> List[str]:
    return [req.replace("\n", "") for req in open(filepath).readlines()]


setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt') # fetching packages from requirements.txt
)
