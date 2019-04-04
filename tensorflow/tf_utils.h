///@file tf_utils.h
///@brief utility functions for tensorflow to optox
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 04.2019

#pragma once

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"
#include "tensor/h_tensor.h"
#include "tensor/d_tensor.h"

// wrappers for iu
template <typename T, unsigned int N>
std::unique_ptr<optox::DTensor<T, N>> getDTensorTensorflow(tensorflow::Tensor &tensor)
{
    auto t_tensor = tensor.tensor<T, N>();
    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = t_tensor.dimension(i);
    std::unique_ptr<optox::DTensor<T, N>> p(new optox::DTensor<T, N>(t_tensor.data(), size, true));

    // do not return a copy but rather move its value
    return move(p);
}

template <typename T, unsigned int N>
std::unique_ptr<optox::DTensor<T, N>> getDTensorTensorflow(const tensorflow::Tensor &tensor)
{
    auto t_tensor = tensor.tensor<T, N>();
    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = t_tensor.dimension(i);
    std::unique_ptr<optox::DTensor<T, N>> p(new optox::DTensor<T, N>(const_cast<T*>(t_tensor.data()), size, true));

    // do not return a copy but rather move its value
    return move(p);
}

template <typename T, unsigned int N>
std::unique_ptr<optox::HTensor<T, N>> getHTensorTensorflow(tensorflow::Tensor &tensor)
{
    auto t_tensor = tensor.tensor<T, N>();
    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = t_tensor.dimension(i);
    std::unique_ptr<optox::HTensor<T, N>> p(new optox::HTensor<T, N>(t_tensor.data(), size, true));

    // do not return a copy but rather move its value
    return move(p);
}

template <typename T, unsigned int N>
std::unique_ptr<optox::HTensor<T, N>> getHTensorTensorflow(const tensorflow::Tensor &tensor)
{
    auto t_tensor = tensor.tensor<T, N>();
    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = t_tensor.dimension(i);
    std::unique_ptr<optox::HTensor<T, N>> p(new optox::HTensor<T, N>(const_cast<T*>(t_tensor.data()), size, true));

    // do not return a copy but rather move its value
    return move(p);
}
