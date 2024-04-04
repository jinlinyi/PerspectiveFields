from setuptools import find_packages, setup

setup(
    name="perspective2d",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,  # This line is important!
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.yaml'],
        # And include any *.msg files found in the 'hello' package, too:
        'perspective2d.config': ['*.yaml'],
    },
    install_requires=[
        "albumentations",
        "matplotlib",
        "numpy",
        "omegaconf",
        "opencv-contrib-python",
        "pillow",
        "pyequilib==0.3.0",
        "scikit-learn",
        "scipy",
        "setuptools",
        "timm",
        "torch",
        "torchvision",
        "yacs",
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
