#pragma once


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

using GPUDevice = Eigen::GpuDevice;

template<typename T>
using Tensor2 = tensorflow::TTypes<T,2>;

template<typename T>
using Tensor3 = tensorflow::TTypes<T,3>;

template <typename T>
struct NablaOperatorForward {
  void operator()(const GPUDevice& d, 
	typename Tensor3<T>::Tensor &out, 
	const typename Tensor2<T>::ConstTensor &x);
};

template <typename T>
struct NablaOperatorAdjoint {
  void operator()(const GPUDevice& d, 
	typename Tensor2<T>::Tensor &out, 
	const typename Tensor3<T>::ConstTensor &x);
};
