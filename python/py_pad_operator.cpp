///@file py_pad_operator.cpp
///@brief python wrappers for the pad operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 01.2021

#include <vector>

#include "py_utils.h"
#include "operators/pad_operator.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename T>
py::array forward2d(optox::Pad2dOperator<T> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, 3>(np_input);

    auto out_size = input->size();
    out_size[1] += op.paddingY();
    out_size[2] += op.paddingX();

    optox::DTensor<T, 3> output(out_size);

    op.forward({&output}, {input.get()});

    return dTensorToNp<T, 3>(output);
}

template<typename T>
py::array adjoint2d(optox::Pad2dOperator<T> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, 3>(np_input);

    auto out_size = input->size();
    out_size[1] -= op.paddingY();
    out_size[2] -= op.paddingX();

    optox::DTensor<T, 3> output(out_size);
    
    op.adjoint({&output}, {input.get()});

    return dTensorToNp<T, 3>(output);
}

template<typename T>
py::array forward3d(optox::Pad3dOperator<T> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, 4>(np_input);

    auto out_size = input->size();
    out_size[1] += op.paddingZ();
    out_size[2] += op.paddingY();
    out_size[3] += op.paddingX();

    optox::DTensor<T, 4> output(out_size);

    op.forward({&output}, {input.get()});

    return dTensorToNp<T, 4>(output);
}

template<typename T>
py::array adjoint3d(optox::Pad3dOperator<T> &op, py::array np_input)
{
    // parse the input tensors
    auto input = getDTensorNp<T, 4>(np_input);

    auto out_size = input->size();
    out_size[1] -= op.paddingZ();
    out_size[2] -= op.paddingY();
    out_size[3] -= op.paddingX();

    optox::DTensor<T, 4> output(out_size);
    
    op.adjoint({&output}, {input.get()});

    return dTensorToNp<T, 4>(output);
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Pad2d_") + typestr;
    py::class_<optox::Pad2dOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int, int, int, int, const std::string&>())
    .def("forward", forward2d<T>)
    .def("adjoint", adjoint2d<T>);

    pyclass_name = std::string("Pad3d_") + typestr;
    py::class_<optox::Pad3dOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int, int, int, int, int, int, const std::string&>())
    .def("forward", forward3d<T>)
    .def("adjoint", adjoint3d<T>);

}

PYBIND11_MODULE(py_pad_operator, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
