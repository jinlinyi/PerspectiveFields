#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name="perspective2d",
    version="1.0",
    author="Linyi Jin",
    description="Code for training Perspective Fields",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["detectron2"],
)