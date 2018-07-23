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

#include "operators/nabla_operator.h"

namespace bp = boost::python;

template<typename T>
PyObject *forward(bp::object &self, bp::object &py_ob_in)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, 2> > iu_in = getLinearDeviceFromNumpy<T, 2>(py_ob_in);

    // allocate the output
    iu::Size<3> out_size({iu_in->size()[0], iu_in->size()[1], 2});
    iu::LinearDeviceMemory<T, 3> iu_out(out_size);

    optox::NablaOperator<T, 2> &op = bp::extract<optox::NablaOperator<T, 2> &>(self);
    op.forward({&iu_out}, {iu_in.get()});

    return iu::python::PyArray_from_LinearDeviceMemory(iu_out);
}

template<typename T>
PyObject *adjoint(bp::object &self, bp::object &py_ob_in)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, 3> > iu_in = getLinearDeviceFromNumpy<T, 3>(py_ob_in);

    // allocate the output
    iu::Size<2> out_size({iu_in->size()[0], iu_in->size()[1]});
    iu::LinearDeviceMemory<T, 2> iu_out(out_size);

    optox::NablaOperator<T, 2> &op = bp::extract<optox::NablaOperator<T, 2> &>(self);
    op.adjoint({&iu_out}, {iu_in.get()});

    return iu::python::PyArray_from_LinearDeviceMemory(iu_out);
}


BOOST_PYTHON_MODULE(PyNablaOperator)
{
  // setup numpy c-api
  import_array();

  // register exceptions
  bp::register_exception_translator<iu::python::Exc>(
      &iu::python::ExcTranslator);

  // basic operator functions
  bp::class_<optox::NablaOperator<float, 2>,
             std::shared_ptr<optox::NablaOperator<float, 2>>,
             boost::noncopyable>("NablaOperator", bp::init<>())
      .def("forward", forward<float>)
      .def("adjoint", adjoint<float>);
}
