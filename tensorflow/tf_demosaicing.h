#pragma once

#include "operators/demosaicing_operator.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

using GPUDevice = Eigen::GpuDevice;

template <typename T>
using Tensor4 = tensorflow::TTypes<T, 4>;

template <typename T>
struct DemosaicingOperatorWrapper
{
  public:
    DemosaicingOperatorWrapper(optox::BayerPattern p) : op(p){};

    void forward(const GPUDevice &d,
                 typename Tensor4<T>::Tensor &out,
                 const typename Tensor4<T>::ConstTensor &in);

    void adjoint(const GPUDevice &d,
                 typename Tensor4<T>::Tensor &out,
                 const typename Tensor4<T>::ConstTensor &in);

  private:
    optox::DemosaicingOperator<T> op;
};
