///@file th_act_operator.h
///@brief PyTorch wrappers for activation functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include <vector>

#include "th_utils.h"
#include "operators/activations/act_rbf.h"
#include "operators/activations/act_linear.h"

#include <torch/torch.h>
#include <pybind11/pybind11.h>


template<typename T>
std::unique_ptr<optox::IActOperator<T>> getOperatorByType(const std::string &type, T v_min, T v_max)
{
    if (type == "rbf")
    {
        std::unique_ptr<optox::IActOperator<T>> p(new optox::RBFActOperator<T>(v_min, v_max));
        return std::move(p);
    }
    else if (type == "linear")
    {
        std::unique_ptr<optox::IActOperator<T>> p(new optox::LinearActOperator<T>(v_min, v_max));
        return std::move(p);
    }
    else
        throw std::runtime_error("Unupported base function type '" + type + "'!");
}

template<typename T>
at::Tensor act_fwd(at::Tensor th_input, at::Tensor th_weights, 
    const std::string &type, 
    T v_min, T v_max)
{
    // parse the input tensors
    auto iu_input = getLinearDeviceTorch<T, 2>(th_input);
    auto iu_weights = getLinearDeviceTorch<T, 2>(th_weights);

    // allocate the output tensor
    auto th_output = at::zeros_like(th_input);
    auto iu_output = getLinearDeviceTorch<T, 2>(th_output);
    
    auto op = getOperatorByType<T>(type, v_min, v_max);
    op->forward({iu_output.get()}, {iu_input.get(), iu_weights.get()});

    return th_output;
}


at::Tensor act_fwd_wrapper(at::Tensor input, at::Tensor weights, 
    const std::string &type,
    float v_min, float v_max)
{
    switch (input.type().scalarType())
    {
        case at::ScalarType::Double:
            return act_fwd<double>(input, weights, type, v_min, v_max);
        case at::ScalarType::Float:
            return act_fwd<float>(input, weights, type, v_min, v_max);
        default:
            throw std::runtime_error("Invalid tensor dtype!");
    }
}

template<typename T>
std::vector<at::Tensor> act_bwd(at::Tensor th_input, at::Tensor th_weights, at::Tensor th_grad_out, 
    const std::string & type,
    T v_min, T v_max)
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
    
    auto op = getOperatorByType<T>(type, v_min, v_max);
    op->adjoint({iu_grad_in.get(), iu_grad_weights.get()}, {iu_input.get(), iu_weights.get(), iu_grad_out.get()});

    return {th_grad_in, th_grad_weights};
}


std::vector<at::Tensor> act_bwd_wrapper(at::Tensor input, at::Tensor weights, at::Tensor grad_out, 
    const std::string &type,
    float v_min, float v_max)
{
    switch (input.type().scalarType())
    {
        case at::ScalarType::Double:
            return act_bwd<double>(input, weights, grad_out, type, v_min, v_max);
        case at::ScalarType::Float:
            return act_bwd<float>(input, weights, grad_out, type, v_min, v_max);
        default:
            throw std::runtime_error("Invalid tensor dtype!");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("act_fwd", &act_fwd_wrapper, "trainable activation function forward implementation");
    m.def("act_bwd", &act_bwd_wrapper, "trainable activation function backward implementation");
}
