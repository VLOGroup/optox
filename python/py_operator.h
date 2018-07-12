///@file py_operator.h
///@brief Interface for operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#define BOOST_PYTHON_STATIC_LIB

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <boost/python/dict.hpp>

#include "operators/ioperator.h"

namespace bp = boost::python;

void mapFromPyObject(const bp::object &py_ob, optox::OperatorConfigDict &out_map)
{
  bp::dict py_dict = bp::extract<bp::dict>(py_ob);
  boost::python::list keys = py_dict.keys();
  for (int i = 0; i < bp::len(keys); ++i)
  {
    std::string key = std::string(bp::extract<const char *>(keys[i]));
    bp::object py_val_fun = py_dict[key].attr("__str__");
    bp::object py_val = py_val_fun();
    std::string value = std::string(bp::extract<const char *>(py_val));
    out_map[key] = value;
  }
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void appendInput(bp::object &self, bp::object &py_arr)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  // here we need to copy the data since its lifetime continues
  op.template appendInput<T, N>(host_mem);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void setInput(bp::object &self, bp::object &py_ob, bp::object &py_arr)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  int index = bp::extract<int>(py_ob);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  op.template setInput<T, N>(index, host_mem);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void appendOutput(bp::object &self, bp::object &py_arr)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  op.template appendOutput<T, N>(host_mem);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
PyObject *getOutput(bp::object &self, bp::object &py_ob)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  int index = bp::extract<int>(py_ob);
  iu::LinearDeviceMemory<T, N> *output = op.template getOutput<T, N>(index);

  return  iu::python::PyArray_from_LinearDeviceMemory(*output);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void setConfig(bp::object &self, bp::object &py_ob)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  optox::OperatorConfigDict config;
  mapFromPyObject(py_ob, config);
  op.setConfig(config);
}