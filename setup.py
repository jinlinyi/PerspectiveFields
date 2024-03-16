from setuptools import find_packages, setup

setup(
    name="perspective2d",
    version="0.9.0",
    packages=find_packages(),
    install_requires=[
        "torchvision",
        "openexr",
        "opencv-contrib-python",
        "albumentations",
        "pyequilib==0.3.0",
        "skylibs",
        "timm",
        "h5py",
        "tensorboard",
        "setuptools==59.5.0",
    ],
    author="Linyi Jin",
    author_email="jinlinyi@umich.edu",
    description="Code for CVPR 2023 Perspective Fields for Single Image Camera Calibration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jinlinyi/PerspectiveFields",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
)
