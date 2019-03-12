///@file py_nabla_operator.cpp
///@brief python wrappers for the nabla operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.07.2018

#include <vector>

#include "py_utils.h"
#include "operators/nabla_operator.h"
#include "operators/nabla2_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T, int N>
py::array forward(optox::NablaOperator<T, N> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, N>(np_input);

    optox::Shape<N+1> out_size;
    for (unsigned int i = 0; i < N; ++i)
        out_size[i] = input->size()[i];
    out_size[N] = N;
    optox::DTensor<T, N+1> output(out_size);

    op.forward({&output}, {input.get()});

    return dTensorToNp<T, N+1>(output);
}

template<typename T, int N>
py::array adjoint(optox::NablaOperator<T, N> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, N+1>(np_input);

    optox::Shape<N> out_size;
    for (unsigned int i = 0; i < N; ++i)
        out_size[i] = input->size()[i];
    optox::DTensor<T, N> output(out_size);
    
    op.adjoint({&output}, {input.get()});

    return dTensorToNp<T, N>(output);
}

template<typename T, int N>
py::array forward2(optox::Nabla2Operator<T, N> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, N+1>(np_input);

    optox::Shape<N+1> out_size;
    for (unsigned int i = 0; i < N; ++i)
        out_size[i] = input->size()[i];
    out_size[N] = N*N;
    optox::DTensor<T, N+1> output(out_size);

    op.forward({&output}, {input.get()});

    return dTensorToNp<T, N+1>(output);
}

template<typename T, int N>
py::array adjoint2(optox::Nabla2Operator<T, N> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, N+1>(np_input);

    optox::Shape<N+1> out_size;
    for (unsigned int i = 0; i < N; ++i)
        out_size[i] = input->size()[i];
    out_size[N] = N;
    optox::DTensor<T, N+1> output(out_size);

    op.adjoint({&output}, {input.get()});

    return dTensorToNp<T, N+1>(output);
}

template<typename T, int N>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Nabla_") + std::to_string(N) + "d_" + typestr;
    py::class_<optox::NablaOperator<T, N>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward<T, N>)
    .def("adjoint", adjoint<T, N>);

    pyclass_name = std::string("Nabla2_") + std::to_string(N) + "d_" + typestr;
    py::class_<optox::Nabla2Operator<T, N>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward2<T, N>)
    .def("adjoint", adjoint2<T, N>);
}

PYBIND11_MODULE(py_nabla_operator, m)
{
    declare_op<float, 2>(m, "float");
    declare_op<double, 2>(m, "double");

    declare_op<float, 3>(m, "float");
    declare_op<double, 3>(m, "double");
}
