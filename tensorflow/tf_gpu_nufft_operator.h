#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"

using GPUDevice = Eigen::GpuDevice;

struct applyGpuNufftForwardOperator {
  void operator()(const GPUDevice& d, 
	typename tensorflow::TTypes<tensorflow::complex64,3>::Tensor &rawdata, 
	const typename tensorflow::TTypes<tensorflow::complex64,3>::ConstTensor &img, 
	const typename tensorflow::TTypes<tensorflow::complex64,4>::ConstTensor &sensitivities,
    const typename tensorflow::TTypes<float,3>::ConstTensor &trajectory,
    const typename tensorflow::TTypes<float,2>::ConstTensor &dcf,
    const int& osf,
    const int& sector_width,
    const int& kernel_width,
    const int& img_dim);
};

struct applyGpuNufftAdjointOperator {
  void operator()(const GPUDevice& d, 
	typename tensorflow::TTypes<tensorflow::complex64,3>::Tensor &img, 
	const typename tensorflow::TTypes<tensorflow::complex64,3>::ConstTensor &rawdata, 
	const typename tensorflow::TTypes<tensorflow::complex64,4>::ConstTensor &sensitivities,
    const typename tensorflow::TTypes<float,3>::ConstTensor &trajectory,
    const typename tensorflow::TTypes<float,2>::ConstTensor &dcf,
    const int& osf,
    const int& sector_width,
    const int& kernel_width,
    const int& img_dim);
};