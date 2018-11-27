#pragma once

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

template<typename T>
using Tensor3 = tensorflow::TTypes<T,3>;

namespace tficg {
  enum borderMode_t
  {
    BORDER_MODE_INVALID = -1,
    BORDER_MODE_ZERO = 0,
    BORDER_MODE_REPLICATE = 1,
    BORDER_MODE_SYMMETRIC = 2,
    BORDER_MODE_VALID = 3,
    BORDER_MODE_EDGESYMMETRIC = 4,
  };

  static borderMode_t strToBorderMode(const std::string &str)
  {

    if (str.compare("REPLICATE") == 0)
      return BORDER_MODE_REPLICATE;
    else if (str.compare("SYMMETRIC") == 0)
      return BORDER_MODE_SYMMETRIC;
//    else if (str.compare("ZERO") == 0)
//      return BORDER_MODE_ZERO;
//    else if (str.compare("VALID") == 0)
//      return BORDER_MODE_VALID;
//    else if (str.compare("EDGESYMMETRIC") == 0)
//      return BORDER_MODE_EDGESYMMETRIC;
    else
      return BORDER_MODE_INVALID;
  }
}
