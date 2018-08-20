// tf_filters.h

#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tficg {
  enum interpolation_t
  {
    INTERPOLATE_INVALID = -1,
    INTERPOLATE_BILINEAR = 0,
    INTERPOLATE_BICUBIC = 1,
  };

  static interpolation_t strToInterpolation(const std::string &str)
  {

    if (str.compare("BILINEAR") == 0)
      return INTERPOLATE_BILINEAR;
    else if (str.compare("BICUBIC") == 0)
      return INTERPOLATE_BICUBIC;
    else
      return INTERPOLATE_INVALID;
  }
}


template<typename T>
using Tensor1 = tensorflow::TTypes<T,1>;
template<typename T>
using Tensor4 = tensorflow::TTypes<T,4>;
template<typename T>
using Tensor5 = tensorflow::TTypes<T,5>;

// Radial basis function Activation Functor ------------------------------------
template<typename Device, typename T, tficg::interpolation_t>
struct RotateFilterFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor4<T>::ConstTensor &x,
                  const typename Tensor1<T>::ConstTensor &angles,
                  typename Tensor5<T>::Tensor &out);
};
template<typename Device, typename T, tficg::interpolation_t>
struct RotateFilterGradFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor1<T>::ConstTensor &angles,
                  const typename Tensor5<T>::ConstTensor &grad_out,
                  typename Tensor4<T>::Tensor &grad_x);
};
// -----------------------------------------------------------------------------
