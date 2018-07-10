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

#include "operators/addoperator.h"

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
    // unsigned int key = bp::extract<unsigned int>(keys[i]);
    // out_map[key] = bp::extract<double>(py_dict[key]);
  }
}

template<template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void appendInput(bp::object& self, bp::object& py_arr)
{
  TOperator<T, N>& op = bp::extract<TOperator<T, N>&>(self);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  iu::LinearDeviceMemory<T, N> device_mem(host_mem.size());
  iu::copy(&host_mem, &device_mem);
  op.appendInput(device_mem);
}

template<template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void appendOutput(bp::object& self, bp::object& py_arr)
{
  TOperator<T, N>& op = bp::extract<TOperator<T, N>&>(self);
  iu::LinearHostMemory<T, N> host_mem(py_arr);
  iu::LinearDeviceMemory<T, N> device_mem(host_mem.size());
  iu::copy(&host_mem, &device_mem);
  op.appendOutput(device_mem);
}

template<template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void setConfig(bp::object& self, bp::object& py_ob)
{
  TOperator<T, N>& op = bp::extract<TOperator<T, N>&>(self);
  // extract the python dictionary
  optox::OperatorConfigDict config;
  mapFromPyObject(py_ob, config);
  op.setConfig(config);
}

//==============================================================================
// create python module
//==============================================================================

BOOST_PYTHON_MODULE(pyaddoperator)  // name must (!) be the same as the resulting *.so file
// get python ImportError about missing init function otherwise
// probably best to sort it out in cmake...
{
    import_array();                   // initialize numpy c-api
    bp::register_exception_translator<iu::python::Exc>(
        &iu::python::ExcTranslator);

    // // Cartesian MRI operator
    // bp::class_<MriCartesianOperator<InputType, OutputType>,
    //     std::shared_ptr<MriCartesianOperator<InputType, OutputType>>,
    //     boost::noncopyable>("MriCartesianOperator", bp::init<>())
    //     .def(bp::self_ns::str(bp::self))  // allow debug printing
    //     .def("setMask", setMask<MriCartesianOperator, InputType, OutputType>)
    //     .def("setCoilSens", setCoilSens<MriCartesianOperator, InputType, OutputType>)
    //     .def("forward", forward<MriCartesianOperator, InputType, OutputType>)
    //     .def("adjoint", adjoint<MriCartesianOperator, InputType, OutputType>);


    bp::class_<optox::AddOperator<float, 1>,
        std::shared_ptr<optox::AddOperator<float, 1>>,
        boost::noncopyable>("AddOperator", bp::init<>())
        .def("appendInput", appendInput<optox::AddOperator, float, 1>)
        .def("appendOutput", appendOutput<optox::AddOperator, float, 1>)
        .def("setConfig", setConfig<optox::AddOperator, float, 1>)
        .def("apply", &optox::AddOperator<float, 1>::apply);
}
