///@file th_rot_operator.h
///@brief PyTorch wrappers for rotation operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019

#include <vector>

#include "th_utils.h"
#include "operators/rot_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
at::Tensor forward(optox::RotOperator<T> &op, at::Tensor th_input, at::Tensor th_angles)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 4>(th_input);
    auto angles = getDTensorTorch<T, 1>(th_angles);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(input->size()[0]);
    shape.push_back(angles->size()[0]);
    shape.push_back(input->size()[1]);
    shape.push_back(input->size()[2]);
    shape.push_back(input->size()[3]);
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, 5>(th_output);

    op.forward({output.get()}, {input.get(), angles.get()});

    return th_output;
}

template<typename T>
at::Tensor adjoint(optox::RotOperator<T> &op, at::Tensor th_input, at::Tensor th_angles)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 5>(th_input);
    auto angles = getDTensorTorch<T, 1>(th_angles);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(input->size()[0]);
    shape.push_back(input->size()[2]);
    shape.push_back(input->size()[3]);
    shape.push_back(input->size()[4]);
    auto th_output = at::empty(shape, th_input.options());
    auto output = getDTensorTorch<T, 4>(th_output);
    
    op.adjoint({output.get()}, {input.get(), angles.get()});

    return th_output;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Rot_") + typestr;
    py::class_<optox::RotOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
