# copy the libraries
cp ../../lib/tf/TfActivationOperators.so ./optotf/activations/TfActivationOperators.so
cp ../../lib/tf/TfMetamorphosisOperator.so ./optotf/interpolation/TfMetamorphosisOperator.so
cp ../../lib/tf/TfRotateFiltersOperator.so ./optotf/interpolation/TfRotateFiltersOperator.so

# build the package
python setup.py sdist bdist_wheel

# install
#pip install dist/optotf-0.1.dev0-cp36-cp36m-linux_x86_64.whl --upgrade

