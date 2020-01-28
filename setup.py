from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="cvxpyrepair",
    version="0.1",
    description="Automatic Repair of CVXPY problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy >= 1.15",
        "scipy >= 1.1.0",
        "diffcp >= 1.0.13",
        "cvxpy >= 1.1.0a1"],
    license="Apache License, Version 2.0",
    url="https://github.com/cvxgrp/cvxpyrepair",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
