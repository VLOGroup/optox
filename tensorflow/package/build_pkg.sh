#!/bin/sh

RED='\033[1;31m'  # Bold red
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

printf "\n${BLUE}Removing old build artifacts...${NC}\n"
rm -rf build
rm -rf dist
rm -rf tfoptotf.egg-info

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
printf "\n${BLUE}copying '.so' libraries into the package/optotf/ folder...${NC}\n"
check_command cp ../../lib/tf/TfActivationOperators.so ./optotf/activations/TfActivationOperators.so
check_command cp ../../lib/tf/TfMetamorphosisOperator.so ./optotf/interpolation/TfMetamorphosisOperator.so
check_command cp ../../lib/tf/TfRotateFiltersOperator.so ./optotf/interpolation/TfRotateFiltersOperator.so

# build the package
printf "\n${BLUE}Building the Wheel Package...${NC}\n"
check_command  python setup.py sdist bdist_wheel


# install
#pip install dist/optotf-0.1.dev0-cp36-cp36m-linux_x86_64.whl --upgrade
printf "\nDone, install the following wheel package by using \n"
printf "   ${GREEN}pip install --upgrade   "
ls dist/*.whl
printf "${NC} \n"
