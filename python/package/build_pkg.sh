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
printf "\n${GREEN}copying '.so' libraries into the package/optotf/ folder...${NC}\n"
check_command cp ../../lib/python/PyNablaOperator.so ./optopy/nabla/PyNablaOperator.so

# build the package
printf "\n${GREEN}Building the Wheel Package...${NC}\n"
check_command  python setup.py sdist bdist_wheel

# install
printf "\nDone, install the following wheel package by using \n"
printf "   ${GREEN}pip install --upgrade   "
ls dist/*.whl
printf "${NC} \n"
