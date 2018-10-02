import setuptools
from setuptools.dist import Distribution
import distutils
import tensorflow as tf
import subprocess

# If precompiled tensorflow isused, one has to destinguish between "tensorflow" and "tensorflow-gpu"
tfCPU = not subprocess.call(["pip","-q","show","tensorflow"] )
tfGPU = not subprocess.call(["pip","-q","show","tensorflow-gpu"] )
if tfCPU:
  tfstr = "tensorflow == {}".format(tf.VERSION)
if tfGPU:
  tfstr = "tensorflow-gpu == {}".format(tf.VERSION)
if (tfGPU and tfCPU) or not (tfGPU or tfCPU):
  tfstr = ""
  assert False, "\n\nunexpected error, is tensorflow or tensorflow-gpu installed with pip?\n\n"
  exit(1)
print ("=>required tensorflow for pip: %s\n"% tfstr)


# define requirements
REQUIRED_PACKAGES = [
    tfstr, # tensorflow or tensorflow-gpu
]

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

with open("README.md", "r") as fh:
    long_description = fh.read()
    setuptools.setup(
        name="optotf",
        version="0.1dev",
        author="Erich Kobler",
        author_email="erich.kobler@icg.tugraz.at",
        description="TensorFlow wrappers for optoX library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        # url="https://github.com/pypa/sampleproject",
        platform=distutils.util.get_platform(),
        packages=setuptools.find_packages(),
        include_package_data=True,
        package_data={'optotf': ["activations/TfActivationOperators.so",
                                 "interpolation/TfRotateFiltersOperator.so",
                                 "interpolation/TfMetamorphosisOperator.so",
                                 "TfNablaOperator.so",
                                 "fft/TfFftOperators.so",
                                 "demosaicing/TFDemosaicingOperator.so"]},
        install_requires=REQUIRED_PACKAGES,
        distclass=BinaryDistribution,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )
