#pragma once 

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

template<typename T>
using Tensor4 = typename tensorflow::TTypes<T,4>;


template<typename Device, typename T>
struct FilterexamplesMedianFilter4dFunctor {
  void operator()(tensorflow::OpKernelContext* context, 
    const typename tensorflow::TTypes<T,4>::ConstTensor &img,
    typename tensorflow::TTypes<T,4>::Tensor &out, 
    int filtersize, int filtertype, bool debug_indices)
    ;
  
  };



template <typename Device, typename T>
struct FilterexamplesMedianFilter4dGradientFunctor{
  void operator()(tensorflow::OpKernelContext* context, 
  const typename tensorflow::TTypes<T,4>::ConstTensor &img,
  const typename tensorflow::TTypes<T,4>::ConstTensor &gradin,
  typename tensorflow::TTypes<T,4>::Tensor &gradout, 
  int filtersize, int filtertype, bool debug_indices)
  ;
};


namespace medianfilter {
  enum filtertype_t
  {
    FILTER_TYPE_INVALID = -1,
    FILTER_TYPE_SIMPLE = 0,
    FILTER_TYPE_SHAREDMEMORY = 1,
  };

  static filtertype_t strToFiltertype(const std::string &str)
  {

    if (str.compare("SIMPLE") == 0)
      return FILTER_TYPE_SIMPLE;
    else if (str.compare("SHAREDMEMORY") == 0)
      return FILTER_TYPE_SHAREDMEMORY;
    else
      return FILTER_TYPE_INVALID;
  }
}



