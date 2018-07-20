#pragma once


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

using GPUDevice = Eigen::GpuDevice;

template<typename T>
using Tensor1 = tensorflow::TTypes<T,1>;

template <typename T>
struct AddOperatorForward {
  void operator()(const GPUDevice& d, 
	typename Tensor1<T>::Tensor &out, 
	const typename Tensor1<T>::ConstTensor &in_1, 
	const typename Tensor1<T>::ConstTensor &in_2);
};

template <typename T>
struct AddOperatorAdjoint {
  void operator()(const GPUDevice& d, 
	typename Tensor1<T>::Tensor &out_1, 
	typename Tensor1<T>::Tensor &out_2, 
	const typename Tensor1<T>::ConstTensor &in_);
};
