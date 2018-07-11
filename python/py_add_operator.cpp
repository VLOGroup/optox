///@file pyMriReconstruction.cpp
///@brief Python wrapper for MRI reconstruction tools
///@author Kerstin Hammernik <hammernik@icg.tugraz.at>
///@date 01.09.2016

#include <iostream>
#define BOOST_PYTHON_STATIC_LIB

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <boost/python/dict.hpp>

#include "operators/add_operator.h"

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
  iu::LinearDeviceMemory<T, N> device_mem(host_mem.size());
  iu::copy(&host_mem, &device_mem);
  op.template appendInput<T, N>(device_mem, true);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void setInput(bp::object &self, bp::object &py_ob, bp::object &py_arr)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  int index = bp::extract<int>(py_ob);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  iu::LinearDeviceMemory<T, N> device_mem(host_mem.size());
  iu::copy(&host_mem, &device_mem);
  op.template setInput<T, N>(index, device_mem);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void appendOutput(bp::object &self, bp::object &py_arr)
{
  TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  iu::LinearDeviceMemory<T, N> device_mem(host_mem.size());
  iu::copy(&host_mem, &device_mem);
  op.template appendOutput<T, N>(device_mem, true);
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

//==============================================================================
// create python module
//==============================================================================

BOOST_PYTHON_MODULE(PyAddOperator) // name must (!) be the same as the resulting *.so file
// get python ImportError about missing init function otherwise
// probably best to sort it out in cmake...
{
  import_array();
  bp::register_exception_translator<iu::python::Exc>(
      &iu::python::ExcTranslator);

  bp::class_<optox::AddOperator<float, 1>,
             std::shared_ptr<optox::AddOperator<float, 1>>,
             boost::noncopyable>("AddOperator", bp::init<>())
      .def("append_input", appendInput<optox::AddOperator, float, 1>)
      .def("set_input", setInput<optox::AddOperator, float, 1>)
      .def("append_output", appendOutput<optox::AddOperator, float, 1>)
      .def("get_output", getOutput<optox::AddOperator, float, 1>)
      .def("set_config", setConfig<optox::AddOperator, float, 1>)
      .def("apply", &optox::AddOperator<float, 1>::apply);
}
