///@file py_add_operator.cpp
///@brief python wrappers for the basic add operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.07.2018

#include <iostream>
#define BOOST_PYTHON_STATIC_LIB

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <boost/python/tuple.hpp>

#include "py_utils.h"

#include "operators/add_operator.h"

namespace bp = boost::python;

template<typename T, unsigned int N>
PyObject *forward(bp::object &self, bp::object &py_ob_in1, bp::object &py_ob_in2)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, N> > iu_in1 = getLinearDeviceFromNumpy<T, N>(py_ob_in1);
    std::unique_ptr<iu::LinearDeviceMemory<T, N> > iu_in2 = getLinearDeviceFromNumpy<T, N>(py_ob_in2);

    // allocate the output
    iu::LinearDeviceMemory<T, N> iu_out(iu_in1->size());

    optox::AddOperator<T, N> &op = bp::extract<optox::AddOperator<T, N> &>(self);
    op.forward({&iu_out}, {iu_in1.get(), iu_in2.get()});

    return iu::python::PyArray_from_LinearDeviceMemory(iu_out);
}

template<typename T, unsigned int N>
bp::object adjoint(bp::object &self, bp::object &py_ob_in)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, N> > iu_in = getLinearDeviceFromNumpy<T, N>(py_ob_in);

    // allocate the output
    iu::LinearDeviceMemory<T, N> iu_out1(iu_in->size());
    iu::LinearDeviceMemory<T, N> iu_out2(iu_in->size());

    optox::AddOperator<T, N> &op = bp::extract<optox::AddOperator<T, N> &>(self);
    op.adjoint({&iu_out1, &iu_out2}, {iu_in.get()});

    return bp::make_tuple(
        bp::handle<>(iu::python::PyArray_from_LinearDeviceMemory(iu_out1)),
        bp::handle<>(iu::python::PyArray_from_LinearDeviceMemory(iu_out2))
    );
}


BOOST_PYTHON_MODULE(PyAddOperator)
{
  // setup numpy c-api
  import_array();

  // register exceptions
  bp::register_exception_translator<iu::python::Exc>(
      &iu::python::ExcTranslator);

  // basic operator functions
  bp::class_<optox::AddOperator<float, 1>,
             std::shared_ptr<optox::AddOperator<float, 1>>,
             boost::noncopyable>("AddOperator", bp::init<>())
      .def("set_config", setConfig<optox::AddOperator, float, 1>)
      .def("forward", forward<float, 1>)
      .def("adjoint", adjoint<float, 1>);
}
