// tf_metamorphosis.h

#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tf_filters.h"


template<typename T>
using Tensor4 = tensorflow::TTypes<T,4>;
template<typename T>
using Tensor5 = tensorflow::TTypes<T,5>;

template<typename Device, typename T, tficg::interpolation_t>
struct MetamorphosisInterpolationFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor5<T>::ConstTensor &x,
				          const typename Tensor4<T>::ConstTensor &phi,
                  typename Tensor5<T>::Tensor &out);
};
template<typename Device, typename T, tficg::interpolation_t>
struct MetamorphosisInterpolationGradFunctor {
  void operator()(tensorflow::OpKernelContext *context,
		  	  	      const typename Tensor5<T>::ConstTensor &x,
				          const typename Tensor4<T>::ConstTensor &phi,
                  const typename Tensor5<T>::ConstTensor &grad_out,
                  typename Tensor5<T>::Tensor &grad_x,
				          typename Tensor4<T>::Tensor &grad_phi);
};

template<typename Device, typename T, tficg::interpolation_t>
struct InterpolationFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor4<T>::ConstTensor &x,
				          const typename Tensor4<T>::ConstTensor &phi,
                  typename Tensor4<T>::Tensor &out);
};
template<typename Device, typename T, tficg::interpolation_t>
struct InterpolationGradFunctor {
  void operator()(tensorflow::OpKernelContext *context,
		  	  	      const typename Tensor4<T>::ConstTensor &x,
				          const typename Tensor4<T>::ConstTensor &phi,
                  const typename Tensor4<T>::ConstTensor &grad_out,
                  typename Tensor4<T>::Tensor &grad_x,
				          typename Tensor4<T>::Tensor &grad_phi);
};
