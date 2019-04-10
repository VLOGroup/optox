///@file th_demosaicing_operator.h
///@brief PyTorch wrappers for demosaicing operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 04.2019

#include <vector>

#include "th_utils.h"
#include "operators/demosaicing_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
at::Tensor forward(optox::DemosaicingOperator<T> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 4>(th_input);

    // allocate the output tensor
    auto in_shape = th_input.sizes().vec();
    in_shape[3] = 1;
    auto th_output = at::empty(in_shape, th_input.type());
    auto output = getDTensorTorch<T, 4>(th_output);

    op.forward({output.get()}, {input.get()});

    return th_output;
}

template<typename T>
at::Tensor adjoint(optox::DemosaicingOperator<T> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto input = getDTensorTorch<T, 4>(th_input);

    // allocate the output tensor
    auto in_shape = th_input.sizes().vec();
    in_shape[3] = 3;
    auto th_output = at::empty(in_shape, th_input.type());
    auto output = getDTensorTorch<T, 4>(th_output);
    
    op.adjoint({output.get()}, {input.get()});

    return th_output;
}

template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("Demosaicing_") + typestr;
    py::class_<optox::DemosaicingOperator<T>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<const std::string&>())
    .def("forward", forward<T>)
    .def("adjoint", adjoint<T>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
