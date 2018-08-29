///@file py_nabla_operator.cpp
///@brief python wrappers for the nabla operator
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

template <typename T, unsigned int N>
PyObject *forward(bp::object &self, bp::object &py_ob_in)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, N>> iu_in = getLinearDeviceFromNumpy<T, N>(py_ob_in);

    // allocate the output
    iu::Size<N+1> out_size;
    for (int i = 0; i < N; ++i)
        out_size[i] = iu_in->size()[i];
    out_size[N] = N;
    iu::LinearDeviceMemory<T, N+1> iu_out(out_size);

    optox::NablaOperator<T, N> &op = bp::extract<optox::NablaOperator<T, N> &>(self);
    op.forward({&iu_out}, {iu_in.get()});

    return iu::python::PyArray_from_LinearDeviceMemory(iu_out);
}

template <typename T, unsigned int N>
PyObject *adjoint(bp::object &self, bp::object &py_ob_in)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, N+1>> iu_in = getLinearDeviceFromNumpy<T, N+1>(py_ob_in);

    // allocate the output
    iu::Size<N> out_size;
    for (int i = 0; i < N; ++i)
        out_size[i] = iu_in->size()[i];
    iu::LinearDeviceMemory<T, N> iu_out(out_size);

    optox::NablaOperator<T, N> &op = bp::extract<optox::NablaOperator<T, N> &>(self);
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
        .def("forward", forward<float, 2>)
        .def("adjoint", adjoint<float, 2>);

    bp::class_<optox::NablaOperator<float, 3>,
               std::shared_ptr<optox::NablaOperator<float, 3>>,
               boost::noncopyable>("Nabla3Operator", bp::init<>())
        .def("forward", forward<float, 3>)
        .def("adjoint", adjoint<float, 3>);
}
