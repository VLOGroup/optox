///@file py_act_operator.cpp
///@brief python wrappers for activation operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include <iostream>
#define BOOST_PYTHON_STATIC_LIB

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <boost/python/tuple.hpp>

#include "py_utils.h"

#include "operators/activations/act_rbf.h"

namespace bp = boost::python;

template <typename T>
PyObject *forward(bp::object &self,
                  bp::object &py_ob_in1, bp::object &py_ob_in2)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, 2>> iu_input = getLinearDeviceFromNumpy<T, 2>(py_ob_in1);
    std::unique_ptr<iu::LinearDeviceMemory<T, 2>> iu_weights = getLinearDeviceFromNumpy<T, 2>(py_ob_in2);

    // allocate the output
    iu::LinearDeviceMemory<T, 2> iu_output(iu_input->size());

    optox::IActOperator<T> &op = bp::extract<optox::IActOperator<T> &>(self);
    op.forward({&iu_output}, {iu_input.get(), iu_weights.get()});

    return iu::python::PyArray_from_LinearDeviceMemory(iu_output);
}

template <typename T>
PyObject *adjoint(bp::object &self,
                  bp::object &py_ob_in1, bp::object &py_ob_in2, bp::object &py_ob_in3)
{
    std::unique_ptr<iu::LinearDeviceMemory<T, 2>> iu_input = getLinearDeviceFromNumpy<T, 2>(py_ob_in1);
    std::unique_ptr<iu::LinearDeviceMemory<T, 2>> iu_weights = getLinearDeviceFromNumpy<T, 2>(py_ob_in2);
    std::unique_ptr<iu::LinearDeviceMemory<T, 2>> iu_grad_output = getLinearDeviceFromNumpy<T, 2>(py_ob_in3);

    // allocate the output
    iu::LinearDeviceMemory<T, 2> iu_grad_input(iu_input->size());
    iu::LinearDeviceMemory<T, 2> iu_grad_weights(iu_weights->size());

    optox::IActOperator<T> &op = bp::extract<optox::IActOperator<T> &>(self);
    op.adjoint({&iu_grad_input, &iu_grad_weights}, {iu_input.get(), iu_weights.get(), iu_grad_output.get()});

    return  bp::make_tuple(bp::object(bp::handle<>(iu::python::PyArray_from_LinearDeviceMemory(iu_grad_input))),
                           bp::object(bp::handle<>(iu::python::PyArray_from_LinearDeviceMemory(iu_grad_input)))).ptr(); 
}

BOOST_PYTHON_MODULE(PyActOperator)
{
    // setup numpy c-api
    import_array();

    // register exceptions
    bp::register_exception_translator<iu::python::Exc>(
        &iu::python::ExcTranslator);

    // Gaussian radial basis functions
    bp::class_<optox::RBFActOperator<float>,
               std::shared_ptr<optox::RBFActOperator<float>>,
               boost::noncopyable>("RBFActOperator_float", bp::init<float, float>())
        .def("forward", forward<float>)
        .def("adjoint", adjoint<float>);

    bp::class_<optox::RBFActOperator<double>,
             std::shared_ptr<optox::RBFActOperator<double>>,
             boost::noncopyable>("RBFActOperator_double", bp::init<double, double>())
      .def("forward", forward<double>)
      .def("adjoint", adjoint<double>);
}
