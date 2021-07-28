///@file th_nabla_operator.h
///@brief PyTorch wrappers for warp operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019

#include <vector>

#include "th_utils.h"
#include "operators/warp_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
at::Tensor forward(optox::WarpOperator<T> &op, at::Tensor th_x, at::Tensor th_u)
{
    // parse the tensors
    auto x = getDTensorTorch<T, 4>(th_x);
    auto u = getDTensorTorch<T, 4>(th_u);

    // allocate the output tensor
    at::Tensor th_out = at::empty_like(th_x);
    auto out = getDTensorTorch<T, 4>(th_out);

    op.forward({out.get()}, {x.get(), u.get()});

    return th_out;
}

template<typename T>
at::Tensor adjoint(optox::WarpOperator<T> &op, at::Tensor th_grad_out, at::Tensor th_u)
{
    // parse the tensors
    auto grad_out = getDTensorTorch<T, 4>(th_grad_out);
    auto u = getDTensorTorch<T, 4>(th_u);

    // allocate the output tensor
    at::Tensor th_grad_x = at::empty_like(th_grad_out);
    auto grad_x = getDTensorTorch<T, 4>(th_grad_x);
    
    op.adjoint({grad_x.get()}, {grad_out.get(), u.get()});

    return th_grad_x;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Warp_") + typestr;
    py::class_<optox::WarpOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
