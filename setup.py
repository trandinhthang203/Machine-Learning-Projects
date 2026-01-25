from setuptools import find_packages, setup
from typing import List

HYTHON_E_DOT = "-e ."

def get_requirements(filepath) -> List[str]:
    requirements = []

    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [i.replace("\n", "") for i in requirements]

        if HYTHON_E_DOT in requirements:
            requirements.remove(HYTHON_E_DOT)

setup(
    name="King The House Predict",
    version="1.0.0",
    description="End to end ML project House Price",
    author="ThangTD",
    author_email="tranthangkhuong203@gamil.com",
    url="https://www.youtube.com/watch?v=KhTCatAKVpk&ab_channel=JustaTeeMusic",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)