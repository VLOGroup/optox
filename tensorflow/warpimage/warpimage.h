#pragma once 
// => MOVE TO HEADER
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"


template<typename T>
using Tensor1 = tensorflow::TTypes<T,1>;
template<typename T>
using Tensor4 = tensorflow::TTypes<T,4>;
template<typename T>
using Tensor5 = tensorflow::TTypes<T,5>;


namespace tficg {
  enum interp_type_t
  {
    INTERP_TYPE_INVALID = -1,
    INTERP_TYPE_BILINEAR = 0,
    INTERP_TYPE_BICUBIC_2POINTS = 1,
    INTERP_TYPE_BICUBIC_4POINTS = 2,
  };

  static interp_type_t strTointerp_type(const std::string &str)
  {

    if (str.compare("BILINEAR") == 0)
      return INTERP_TYPE_BILINEAR;
    else if (str.compare("BICUBIC_2POINTS") == 0)
      return INTERP_TYPE_BICUBIC_2POINTS;
    else if (str.compare("BICUBIC_4POINTS") == 0)
      return INTERP_TYPE_BICUBIC_4POINTS;
    else
      return INTERP_TYPE_INVALID;
  }
}



// specify the Functor once in general form in the header for CPU & GPU version
template<typename Device, typename T>
struct WarpimageFunctor {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor &flow,
                  typename Tensor4<T>::Tensor &out, tficg::interp_type_t interp_type) ;
};


// specify the Functor once in general form in the header for CPU & GPU version
template<typename Device, typename T>
struct WarpimageGradientsFunctor{
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor  &flow ,
                  const typename Tensor4<T>::ConstTensor  &gradIn ,
                  typename Tensor4<T>::Tensor &gradImg,
                  typename Tensor4<T>::Tensor &gradFlow,
                  int interp_type);
};


