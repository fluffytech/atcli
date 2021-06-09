from setuptools import setup, find_packages
import os


setup(
    name="atcli",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.7.3",
    install_requires=[
        "click==8.0.0",
    ],
    entry_points={
        'console_scripts': [
            'atcli = atcli.cli:main'
        ]
    }
)
