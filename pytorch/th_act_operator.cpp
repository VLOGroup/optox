///@file th_act_operator.h
///@brief PyTorch wrappers for activation functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include <vector>

#include "th_utils.h"

#include "operators/activations/act.h"
#include "operators/activations/act_rbf.h"
#include "operators/activations/act_linear.h"

#include <torch/extension.h>
#include <pybind11/pybind11.h>


template<typename T>
at::Tensor forward(optox::IActOperator<T> &op, at::Tensor th_input, at::Tensor th_weights)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, 2>(th_input);
    auto iu_weights = getLinearDeviceTorch<T, 2>(th_weights);

    // allocate the output tensor
    auto th_output = at::zeros_like(th_input);
    auto iu_output = getLinearDeviceTorch<T, 2>(th_output);
    
    op.forward({iu_output.get()}, {iu_input.get(), iu_weights.get()});

    return th_output;
}

template<typename T>
std::vector<at::Tensor> adjoint(optox::IActOperator<T> &op, at::Tensor th_input, at::Tensor th_weights, at::Tensor th_grad_out)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, 2>(th_input);
    auto iu_weights = getLinearDeviceTorch<T, 2>(th_weights);
    auto iu_grad_out = getLinearDeviceTorch<T, 2>(th_grad_out);

    // allocate the output tensor
    at::Tensor th_grad_in = at::empty_like(th_input);
    auto iu_grad_in = getLinearDeviceTorch<T, 2>(th_grad_in);
    at::Tensor th_grad_weights = at::empty_like(th_weights);
    auto iu_grad_weights = getLinearDeviceTorch<T, 2>(th_grad_weights);
    
    op.adjoint({iu_grad_in.get(), iu_grad_weights.get()}, {iu_input.get(), iu_weights.get(), iu_grad_out.get()});

    return {th_grad_in, th_grad_weights};
}

template<typename T>
class PyIActOperator : public optox::IActOperator<T> {
public:
    /* Inherit the constructors */
    using optox::IActOperator<T>::IActOperator;

    /* Trampoline (need one for each virtual function) */
    void computeForward(optox::OperatorOutputVector &&outputs,
                        const optox::OperatorInputVector &inputs) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            optox::IActOperator<T>,
            computeForward,
            outputs,
            inputs
        );
    }

    void computeAdjoint(optox::OperatorOutputVector &&outputs,
                        const optox::OperatorInputVector &inputs) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            optox::IActOperator<T>,
            computeAdjoint,
            outputs,
            inputs
        );
    }
};


template<typename T>
std::vector<at::Tensor> forward2(optox::IAct2Operator<T> &op, at::Tensor th_input, at::Tensor th_weights)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, 2>(th_input);
    auto iu_weights = getLinearDeviceTorch<T, 2>(th_weights);

    // allocate the output tensor
    auto th_output = at::zeros_like(th_input);
    auto iu_output = getLinearDeviceTorch<T, 2>(th_output);
    auto th_output_prime = at::zeros_like(th_input);
    auto iu_output_prime = getLinearDeviceTorch<T, 2>(th_output_prime);
    
    op.forward({iu_output.get(), iu_output_prime.get()}, {iu_input.get(), iu_weights.get()});

    return {th_output, th_output_prime};
}

template<typename T>
std::vector<at::Tensor> adjoint2(optox::IAct2Operator<T> &op, at::Tensor th_input, at::Tensor th_weights, 
                                 at::Tensor th_grad_out, at::Tensor th_grad_out_prime)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, 2>(th_input);
    auto iu_weights = getLinearDeviceTorch<T, 2>(th_weights);
    auto iu_grad_out = getLinearDeviceTorch<T, 2>(th_grad_out);
    auto iu_grad_out_prime = getLinearDeviceTorch<T, 2>(th_grad_out_prime);

    // allocate the output tensor
    at::Tensor th_grad_in = at::empty_like(th_input);
    auto iu_grad_in = getLinearDeviceTorch<T, 2>(th_grad_in);
    at::Tensor th_grad_weights = at::empty_like(th_weights);
    auto iu_grad_weights = getLinearDeviceTorch<T, 2>(th_grad_weights);
    
    op.adjoint({iu_grad_in.get(), iu_grad_weights.get()}, {iu_input.get(), iu_weights.get(), iu_grad_out.get(), iu_grad_out_prime.get()});

    return {th_grad_in, th_grad_weights};
}

template<typename T>
class PyIAct2Operator : public optox::IAct2Operator<T> {
public:
    /* Inherit the constructors */
    using optox::IAct2Operator<T>::IAct2Operator;

    /* Trampoline (need one for each virtual function) */
    void computeForward(optox::OperatorOutputVector &&outputs,
                        const optox::OperatorInputVector &inputs) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            optox::IAct2Operator<T>,
            computeForward,
            outputs,
            inputs
        );
    }

    void computeAdjoint(optox::OperatorOutputVector &&outputs,
                        const optox::OperatorInputVector &inputs) override
    {
        PYBIND11_OVERLOAD_PURE(
            void,
            optox::IAct2Operator<T>,
            computeAdjoint,
            outputs,
            inputs
        );
    }
};


template<typename T>
void declare_op(py::module &m, const std::string &typestr)
{
    std::string pyclass_name = std::string("IAct") + "_" + typestr;
    py::class_<optox::IActOperator<T>, PyIActOperator<T>>i_act(m, pyclass_name.c_str());
    i_act.def(py::init<T, T>())
        .def("forward", forward<T>)
        .def("adjoint", adjoint<T>);

    pyclass_name = std::string("RbfAct") + "_" + typestr;
    py::class_<optox::RBFActOperator<T>>(m, pyclass_name.c_str(), i_act)
    .def(py::init<T, T>());

    pyclass_name = std::string("LinearAct") + "_" + typestr;
    py::class_<optox::LinearActOperator<T>>(m, pyclass_name.c_str(), i_act)
    .def(py::init<T, T>());

    pyclass_name = std::string("IAct2") + "_" + typestr;
    py::class_<optox::IAct2Operator<T>, PyIAct2Operator<T>>i_act2(m, pyclass_name.c_str());
    i_act2.def(py::init<T, T>())
        .def("forward", forward2<T>)
        .def("adjoint", adjoint2<T>);

    pyclass_name = std::string("RbfAct2") + "_" + typestr;
    py::class_<optox::RBFAct2Operator<T>>(m, pyclass_name.c_str(), i_act2)
    .def(py::init<T, T>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    declare_op<float>(m, "float");
    declare_op<double>(m, "double");
}
