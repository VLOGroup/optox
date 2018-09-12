///@file tf_activations.h
///@brief TF wrappers for activation functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 17.08.2018

#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tficg {
  enum DerivativeOrder
  {
    DO_ZERO,
    DO_FIRST,
    DO_SECOND,
    DO_INT,
  };

  enum SplineOrder
  {
    SO_LINEAR,
    SO_QUADRATIC,
    SO_CUBIC,
  };

  enum BorderMode
  {
    DO_NONE,
    DO_EXTRAPOLATE,
  };
}

using GPUDevice = Eigen::GpuDevice;

template<typename T>
using Tensor2 = tensorflow::TTypes<T,2>;

#define TF_CALL_ICG_REAL_NUMBER_TYPES(m) \
   TF_CALL_float(m) TF_CALL_double(m)

// Radial basis function Activation Functor
template<typename Device, typename T, tficg::DerivativeOrder N>
struct ActivationRBFFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride);
};
template<typename Device, typename T, tficg::DerivativeOrder N>
struct ActivationRBFGradWFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride);
};

// b-spline Activation Functor
template<typename Device, typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride);
};
template<typename Device, typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineGradWFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride);
};

// Linear interpolation Activation Functor
template<typename Device, typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
struct ActivationInterpolateLinearFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride);
};
template<typename Device, typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
struct ActivationInterpolateLinearGradWFunctor {
  void operator()(tensorflow::OpKernelContext *context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride);
};

