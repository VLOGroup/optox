import setuptools
from setuptools.dist import Distribution
import distutils

import numpy as np

# define requirements
REQUIRED_PACKAGES = [
    "numpy == {}".format(np.__version__)
]

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="optopy",
        version="0.1dev",
        author="Erich Kobler",
        author_email="erich.kobler@icg.tugraz.at",
        description="Python wrappers for optoX library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # url="https://github.com/pypa/sampleproject",
        platform=distutils.util.get_platform(),
        packages=setuptools.find_packages(),
        include_package_data=True,
        package_data={'optopy': ["nabla/PyNablaOperator.so",
                                 "ct/PyCBCTOperator.so"]},
        install_requires=REQUIRED_PACKAGES,
        distclass=BinaryDistribution,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
