///@file th_pad2d_operator.h
///@brief PyTorch wrappers for pad2d operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2020

#include <vector>

#include "th_utils.h"
#include "operators/pad_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
at::Tensor forward2d(optox::Pad2dOperator<T> &op, at::Tensor th_x)
{
    // parse the tensors
    auto x = getDTensorTorch<T, 3>(th_x);

    // allocate the output tensor
    auto in_shape = th_x.sizes().vec();
    std::vector<int64_t> shape;
    shape.push_back(in_shape[0]);
    shape.push_back(in_shape[1]+op.paddingY());
    shape.push_back(in_shape[2]+op.paddingX());
    auto th_out = at::empty(shape, th_x.options());
    auto out = getDTensorTorch<T, 3>(th_out);

    op.forward({out.get()}, {x.get()});

    return th_out;
}

template<typename T>
at::Tensor adjoint2d(optox::Pad2dOperator<T> &op, at::Tensor th_grad_out)
{
    // parse the tensors
    auto grad_out = getDTensorTorch<T, 3>(th_grad_out);

    // allocate the output tensor
    auto in_shape = th_grad_out.sizes().vec();
    std::vector<int64_t> shape;
    shape.push_back(in_shape[0]);
    shape.push_back(in_shape[1]-op.paddingY());
    shape.push_back(in_shape[2]-op.paddingX());
    auto th_grad_x = at::empty(shape, th_grad_out.options());
    auto grad_x = getDTensorTorch<T, 3>(th_grad_x);

    op.adjoint({grad_x.get()}, {grad_out.get()});

    return th_grad_x;
}

template<typename T>
at::Tensor forward3d(optox::Pad3dOperator<T> &op, at::Tensor th_x)
{
    // parse the tensors
    auto x = getDTensorTorch<T, 4>(th_x);

    // allocate the output tensor
    auto in_shape = th_x.sizes().vec();
    std::vector<int64_t> shape;
    shape.push_back(in_shape[0]);
    shape.push_back(in_shape[1]+op.paddingZ());
    shape.push_back(in_shape[2]+op.paddingY());
    shape.push_back(in_shape[3]+op.paddingX());

    auto th_out = at::empty(shape, th_x.options());
    auto out = getDTensorTorch<T, 4>(th_out);

    op.forward({out.get()}, {x.get()});

    return th_out;
}

template<typename T>
at::Tensor adjoint3d(optox::Pad3dOperator<T> &op, at::Tensor th_grad_out)
{
    // parse the tensors
    auto grad_out = getDTensorTorch<T, 4>(th_grad_out);

    // allocate the output tensor
    auto in_shape = th_grad_out.sizes().vec();
    std::vector<int64_t> shape;
    shape.push_back(in_shape[0]);
    shape.push_back(in_shape[1]-op.paddingZ());
    shape.push_back(in_shape[2]-op.paddingY());
    shape.push_back(in_shape[3]-op.paddingX());

    auto th_grad_x = at::empty(shape, th_grad_out.options());
    auto grad_x = getDTensorTorch<T, 4>(th_grad_x);

    op.adjoint({grad_x.get()}, {grad_out.get()});

    return th_grad_x;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Pad2d_") + typestr;
    py::class_<optox::Pad2dOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int,int,int,int,const std::string&>())
    .def("forward", forward2d<T>)
    .def("adjoint", adjoint2d<T>);

    std::string pyclass_name_3d = std::string("Pad3d_") + typestr;
    py::class_<optox::Pad3dOperator<T>>(m, pyclass_name_3d.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<int,int,int,int,int,int,const std::string&>())
    .def("forward", forward3d<T>)
    .def("adjoint", adjoint3d<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
