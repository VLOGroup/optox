///@file th_utils.h
///@brief utility functions for torch to optox
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#pragma once

#include <torch/extension.h>
#include "tensor/h_tensor.h"
#include "tensor/d_tensor.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_NOT_CUDA(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

// wrappers for iu
template <typename T, unsigned int N>
std::unique_ptr<optox::DTensor<T, N>> getDTensorTorch(at::Tensor tensor)
{
    // check the tensor
    CHECK_CUDA(tensor);
    CHECK_CONTIGUOUS(tensor);
    AT_ASSERTM(tensor.ndimension() == N, "invalid tensor dimensions. expected " + std::to_string(N) +
                                             " but got " + std::to_string(tensor.ndimension()) + "!");

    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = tensor.size(N - 1 - i);
    std::unique_ptr<optox::DTensor<T, N>> p(new optox::DTensor<T, N>(tensor.data<T>(), size, true));

    // do not return a copy but rather move its value
    return move(p);
}

template <typename T, unsigned int N>
std::unique_ptr<optox::HTensor<T, N>> getHTensorTorch(at::Tensor tensor)
{
    // check the tensor
    CHECK_NOT_CUDA(tensor);
    CHECK_CONTIGUOUS(tensor);
    AT_ASSERTM(tensor.ndimension() == N, "invalid tensor dimensions. expected " + std::to_string(N) +
                                             " but got " + std::to_string(tensor.ndimension()) + "!");

    // wrap the Tensor into a LinearHostMemory
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = tensor.size(N - 1 - i);
    std::unique_ptr<optox::HTensor<T, N>> p(new optox::HTensor<T, N>(tensor.data<T>(), size, true));

    // do not return a copy but rather move its value
    return move(p);
}