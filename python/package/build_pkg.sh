# copy the libraries
cp ../../lib/python/PyNablaOperator.so ./optopy/nabla/PyNablaOperator.so

# build the package
python setup.py sdist bdist_wheel

# install
#pip install dist/optotf-0.1.dev0-cp36-cp36m-linux_x86_64.whl --upgrade

