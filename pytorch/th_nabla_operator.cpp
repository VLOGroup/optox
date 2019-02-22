///@file th_nabla_operator.h
///@brief PyTorch wrappers for nabla operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019

#include <vector>

#include "th_utils.h"
#include "operators/nabla_operator.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T, int N>
at::Tensor forward(optox::NablaOperator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, N>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    shape.push_back(N);
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin(), in_shape.end());
    auto th_output = at::empty(shape, th_input.type());
    auto iu_output = getLinearDeviceTorch<T, N+1>(th_output);

    op.forward({iu_output.get()}, {iu_input.get()});

    return th_output;
}

template<typename T, int N>
at::Tensor adjoint(optox::NablaOperator<T, N> &op, at::Tensor th_input)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, N+1>(th_input);

    // allocate the output tensor
    std::vector<int64_t> shape;
    auto in_shape = th_input.sizes().vec();
    shape.insert(shape.end(), in_shape.begin()+1, in_shape.end());
    auto th_output = at::empty(shape, th_input.type());
    auto iu_output = getLinearDeviceTorch<T, N>(th_output);
    
    op.adjoint({iu_output.get()}, {iu_input.get()});

    return th_output;
}

template<typename T, int N>
void declare_op(py::module &m, const std::string &typestr)
{
    using Class = optox::NablaOperator<T, N>;
    std::string pyclass_name = std::string("Nabla") + std::to_string(N) + "_" + typestr;
    py::class_<Class>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
    .def(py::init<>())
    .def("forward", forward<T, N>)
    .def("adjoint", adjoint<T, N>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float, 2>(m, "float");
    declare_op<double, 2>(m, "double");

    declare_op<float, 3>(m, "float");
    declare_op<double, 3>(m, "double");
}
