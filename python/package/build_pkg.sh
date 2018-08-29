#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

check_command()
{
  COMMAND=$@
  eval "$COMMAND"

  rc=$?
  if [ $rc -ne 0 ]; then
	printf "${RED} ERROR:${NC} $COMMAND\n"
    exit $rc
  fi
}

# copy the libraries
check_command cp ../../lib/python/PyNablaOperator.so ./optopy/nabla/PyNablaOperator.so

# build the package
check_command python setup.py sdist bdist_wheel

# install
#pip install dist/optotf-0.1.dev0-cp36-cp36m-linux_x86_64.whl --upgrade

