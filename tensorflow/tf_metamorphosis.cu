#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

#include "tf_metamorphosis.h"
#include "tf_activations.h"
#include "tf_cuutils.cuh"

#include <iu/iucore.h>


// definition of actual modulo operator
__device__ inline int mod(int a, int b) {
  int r = a % b;
  return r < 0 ? r + b : r;
}

/**
 * perform bilinear interpolation
 */
template <typename T>
__device__ T interpolate_bilinear(const typename Tensor5<T>::ConstTensor& x,
                                  int ids, int idc, T idy, T idx, int idalpha) {
  const int idx_f = floorf(idx);
  const int idy_f = floorf(idy);

  const int idx_c = idx_f + 1;
  const int idy_c = idy_f + 1;

  const T w = idx - idx_f;
  const T h = idy - idy_f;

  T i_ff = 0, i_fc = 0;
  if (idx_f >= 0 && idx_f < x.dimensions()[4]) {
    if (idy_f >= 0 && idy_f < x.dimensions()[3])
      i_ff = x(ids, idc, idalpha, idy_f, idx_f);

    if (idy_c >= 0 && idy_c < x.dimensions()[3])
      i_fc = x(ids, idc, idalpha, idy_c, idx_f);
  }

  T i_cf = 0, i_cc = 0;
  if (idx_c >= 0 && idx_c < x.dimensions()[4]) {
    if (idy_f >= 0 && idy_f < x.dimensions()[3])
      i_cf = x(ids, idc, idalpha, idy_f, idx_c);

    if (idy_c >= 0 && idy_c < x.dimensions()[3])
      i_cc = x(ids, idc, idalpha, idy_c, idx_c);
  }

  T out = (1 - h) * (1 - w) * i_ff;
  out += (1 - h) * w * i_cf;
  out += h * (1 - w) * i_fc;
  out += h * w * i_cc;

  return out;
}

template <typename T>
__device__ T interpolate_cubic(volatile T* in, T idx, int kernel_size) {
  const int idx_f = floorf(idx);
  const int idx_f_1 = idx_f - 1;
  const int idx_c = idx_f + 1;
  const int idx_c_1 = idx_c + 1;

  // get the input values
  T i_f = 0;
  if (idx_f >= 0 && idx_f < kernel_size) i_f = in[idx_f];
  T i_f_1 = 0;
  if (idx_f_1 >= 0 && idx_f_1 < kernel_size) i_f_1 = in[idx_f_1];
  T i_c = 0;
  if (idx_c >= 0 && idx_c < kernel_size) i_c = in[idx_c];
  T i_c_1 = 0;
  if (idx_c_1 >= 0 && idx_c_1 < kernel_size) i_c_1 = in[idx_c_1];

  // determine the coefficients
  const T p_f = i_f;
  const T p_prime_f = (i_c - i_f_1) / 2;
  const T p_c = i_c;
  const T p_prime_c = (i_c_1 - i_f) / 2;

  const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
  const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
  const T c = p_prime_f;
  const T d = p_f;

  const T u = idx - idx_f;

  T out = u * (u * (u * a + b) + c) + d;

  return out;
}

template <typename T>
__device__ T interpolate_bicubic(const typename Tensor5<T>::ConstTensor& x,
                                 int ids, int idc, T idy, T idx, int idalpha) {
  const int idy_f = floorf(idy);
  const int idx_f = floorf(idx);

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          buff_x[dx + 1] = x(ids, idc, idalpha, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

  return out;
}

template <typename T, tficg::interpolation_t I>
__global__ void metamorphosisWarpKernel(
    const typename Tensor5<T>::ConstTensor x,
    const typename Tensor4<T>::ConstTensor phi,
    const int ids, const int idc,
    typename Tensor5<T>::Tensor out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const int idr = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < x.dimensions()[4] && idy < x.dimensions()[3] &&
      idr < x.dimensions()[2]) {
    // first load the displacement
    const T dx = phi(ids, idy, idx, 0);
    const T dy = phi(ids, idy, idx, 1);
    const T beta = phi(ids, idy, idx, 2);

    const int depth = x.dimensions()[2];

    const int beta_offset = floorf(beta);
    const T d = beta - beta_offset;

    const int idalpha = idr + beta_offset;
    const int idalpha_f = mod(idalpha, depth);
    const int idalpha_c = mod(idalpha_f + 1, depth);

    if (depth == 1)
    {
      switch (I) {
        case tficg::INTERPOLATE_BILINEAR:
        {
          out(ids, idc, idr, idy, idx) = interpolate_bilinear(x, ids, idc, idy + dy, idx + dx, 0);
          break;
        }

        case tficg::INTERPOLATE_BICUBIC:
        {
          out(ids, idc, idr, idy, idx) = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, 0);
          break;
        }
      }
    }
    else
    {
      switch (I) {
        case tficg::INTERPOLATE_BILINEAR:
        {
          T val_f = interpolate_bilinear(x, ids, idc, idy + dy, idx + dx, idalpha_f);
          T val_c = interpolate_bilinear(x, ids, idc, idy + dy, idx + dx, idalpha_c);

          out(ids, idc, idr, idy, idx) = val_f * (1 - d) + val_c * d;
          break;
        }

        case tficg::INTERPOLATE_BICUBIC:
        {
          T buff[4];
          const int idalpha_f_1 = mod(idalpha - 1, depth);
          const int idalpha_c_1 = mod(idalpha + 2, depth);

          buff[0] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_f_1);
          buff[1] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_f);
          buff[2] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_c);
          buff[3] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_c_1);
          out(ids, idc, idr, idy, idx) = interpolate_cubic<T>(buff, d + 1, 4);
          break;
        }
      }
    }
  }
}

template <typename T, tficg::interpolation_t I>
struct MetamorphosisInterpolationFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor5<T>::ConstTensor& x,
                  const typename Tensor4<T>::ConstTensor& phi,
                  typename Tensor5<T>::Tensor& out) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    dim3 block_size(16, 16, 1);
    dim3 grid_size(iu::divUp(x.dimensions()[4], block_size.x),
                   iu::divUp(x.dimensions()[3], block_size.y), x.dimensions()[2]);

    for (int ids = 0; ids < x.dimensions()[0]; ++ids) {
      for (int idc = 0; idc < x.dimensions()[1]; ++idc)
        metamorphosisWarpKernel<T, I>
            <<<grid_size, block_size, 0, d.stream()>>>(x, phi, ids, idc, out);
    }
  }
};

#define REGISTER_GPU_FUNCTOR(T)                      \
  template struct MetamorphosisInterpolationFunctor< \
      GPUDevice, T, tficg::INTERPOLATE_BILINEAR>;    \
  template struct MetamorphosisInterpolationFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


template <typename T>
__device__ T interpolate_bilinear(const typename Tensor4<T>::ConstTensor& x,
                                  int ids, int idc, T idy, T idx) {
  const int idx_f = floorf(idx);
  const int idy_f = floorf(idy);

  const int idx_c = idx_f + 1;
  const int idy_c = idy_f + 1;

  const T w = idx - idx_f;
  const T h = idy - idy_f;

  T i_ff = 0, i_fc = 0;
  if (idx_f >= 0 && idx_f < x.dimensions()[3]) {
    if (idy_f >= 0 && idy_f < x.dimensions()[2])
      i_ff = x(ids, idc, idy_f, idx_f);

    if (idy_c >= 0 && idy_c < x.dimensions()[2])
      i_fc = x(ids, idc, idy_c, idx_f);
  }

  T i_cf = 0, i_cc = 0;
  if (idx_c >= 0 && idx_c < x.dimensions()[3]) {
    if (idy_f >= 0 && idy_f < x.dimensions()[2])
      i_cf = x(ids, idc, idy_f, idx_c);

    if (idy_c >= 0 && idy_c < x.dimensions()[2])
      i_cc = x(ids, idc, idy_c, idx_c);
  }

  T out = (1 - h) * (1 - w) * i_ff;
  out += (1 - h) * w * i_cf;
  out += h * (1 - w) * i_fc;
  out += h * w * i_cc;

  return out;
}

template <typename T>
__device__ T interpolate_bicubic(const typename Tensor4<T>::ConstTensor& x,
                                 int ids, int idc, T idy, T idx) {
  const int idy_f = floorf(idy);
  const int idx_f = floorf(idx);

  T buff_y[4];
  T buff_x[4];

  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[2])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
          buff_x[dx + 1] = x(ids, idc, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

  return out;
}

template <typename T, tficg::interpolation_t I>
__global__ void warpKernel(
    const typename Tensor4<T>::ConstTensor x,
    const typename Tensor4<T>::ConstTensor phi,
    const int ids,
    typename Tensor4<T>::Tensor out) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const int idc = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < x.dimensions()[3] && idy < x.dimensions()[2]) {
    // first load the displacement
    const T dx = phi(ids, idy, idx, 0);
    const T dy = phi(ids, idy, idx, 1);

    switch (I) {
      case tficg::INTERPOLATE_BILINEAR:
      {
        out(ids, idc, idy, idx) = interpolate_bilinear(x, ids, idc, idy + dy, idx + dx);
        break;
      }

      case tficg::INTERPOLATE_BICUBIC:
      {
        out(ids, idc, idy, idx) = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx);
        break;
      }
    }
  }
}

template <typename T, tficg::interpolation_t I>
struct InterpolationFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor4<T>::ConstTensor& x,
                  const typename Tensor4<T>::ConstTensor& phi,
                  typename Tensor4<T>::Tensor& out) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    dim3 block_size(8, 8, 1);
    dim3 grid_size(iu::divUp(x.dimensions()[3], block_size.x),
                   iu::divUp(x.dimensions()[2], block_size.y), x.dimensions()[1]);

    for (int ids = 0; ids < x.dimensions()[0]; ++ids)
        warpKernel<T, I><<<grid_size, block_size, 0, d.stream()>>>(x, phi, ids, out);
  }
};

#define REGISTER_GPU_FUNCTOR(T)                      \
  template struct InterpolationFunctor<GPUDevice, T, tficg::INTERPOLATE_BILINEAR>;    \
  template struct InterpolationFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

// gradient operator -----------------------------------------------------------

/**
  * perform bilinear interpolation adjoint
  */
template <typename T>
__device__ T backpolate_bilinear(typename Tensor5<T>::Tensor& grad_x,
                                 T& grad_idx, T& grad_idy,
                                 const typename Tensor5<T>::ConstTensor& x,
                                 T val, int ids, int idc, T idy, T idx, int idalpha) {
  const int idx_f = floorf(idx);
  const int idy_f = floorf(idy);

  const int idx_c = idx_f + 1;
  const int idy_c = idy_f + 1;

  const T w = idx - idx_f;
  const T h = idy - idy_f;

  T i_ff = 0, i_fc = 0;
  if (idx_f >= 0 && idx_f < grad_x.dimensions()[4]) {
    if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_f, idx_f),
                                (1 - h) * (1 - w) * val);
      i_ff = x(ids, idc, idalpha, idy_f, idx_f);
    }

    if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_c, idx_f),
                                h * (1 - w) * val);
      i_fc = x(ids, idc, idalpha, idy_c, idx_f);
    }
  }

  T i_cf = 0, i_cc = 0;
  if (idx_c >= 0 && idx_c < grad_x.dimensions()[4]) {
    if (idy_f >= 0 && idy_f < grad_x.dimensions()[3]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_f, idx_c),
                                (1 - h) * w * val);
      i_cf = x(ids, idc, idalpha, idy_f, idx_c);
    }

    if (idy_c >= 0 && idy_c < grad_x.dimensions()[3]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, idy_c, idx_c),
                                h * w * val);
      i_cc = x(ids, idc, idalpha, idy_c, idx_c);
    }
  }

  grad_idx += ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * val;
  grad_idy += ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * val;

  T out = (1 - h) * (1 - w) * i_ff;
  out += (1 - h) * w * i_cf;
  out += h * (1 - w) * i_fc;
  out += h * w * i_cc;

  return out;
}

/**
 * perform cubic interpolation on the input image i_in given the
 * index (idx,idy)
 */
template<typename T>
__device__ void backpolate_cubic(volatile T *grad_x, T& grad_idx,
   T* in, T error, T idx, int kernel_size)
{
  const int idx_f = floorf(idx);
  const int idx_f_1 = idx_f - 1;
  const int idx_c = idx_f+1;
  const int idx_c_1 = idx_c+1;

  const T u = idx - idx_f;
  const T uu = u*u;
  const T uuu = uu*u;

  // determine the coefficients
  T d_out_d_p_f_1 = -uuu/2 + uu - u/2;
  T d_out_d_p_f = (3*uuu)/2 - (5*uu)/2 + 1;
  T d_out_d_p_c = -(3*uuu)/2 + 2*uu + u/2;
  T d_out_d_p_c_1 = uuu/2 - uu/2;

  T i_f = 0;
  if (idx_f >= 0 && idx_f < kernel_size)
  {
    i_f = in[idx_f];
    grad_x[idx_f] = d_out_d_p_f   * error;
  }
  else
    grad_x[idx_f] = 0;

  T i_f_1 = 0;
  if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
  {
    i_f_1 = in[idx_f_1];
    grad_x[idx_f_1] = d_out_d_p_f_1 * error;
  }
  else
    grad_x[idx_f_1] = 0;

  T i_c = 0;
  if (idx_c >= 0 && idx_c < kernel_size)
  {
    i_c = in[idx_c];
    grad_x[idx_c] = d_out_d_p_c   * error;
  }
  else
    grad_x[idx_c] = 0;

  T i_c_1 = 0;
  if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
  {
    i_c_1 = in[idx_c_1];
    grad_x[idx_c_1] = d_out_d_p_c_1 * error;
  }
  else
    grad_x[idx_c_1] = 0;

  // determine the coefficients
  const T p_f = i_f;
  const T p_prime_f = (i_c - i_f_1) / 2;
  const T p_c = i_c;
  const T p_prime_c = (i_c_1 - i_f) / 2;

  const T a = 2 * p_f - 2 * p_c + p_prime_f + p_prime_c;
  const T b = -3 * p_f + 3 * p_c - 2 * p_prime_f - p_prime_c;
  const T c = p_prime_f;

  grad_idx += (3*uu*a + 2*b*u + c) * error;
}

template<typename T>
__device__ T backpolate_bicubic(typename Tensor5<T>::Tensor& grad_x,
  T& grad_idx, T& grad_idy,
  const typename Tensor5<T>::ConstTensor& x,
  T val, int ids, int idc, T idy, T idx, int idalpha)
{
  const int idy_f = floorf(idy);
  const int idx_f = floorf(idx);

  T buff_y[4];
  T buff_x[4];

  // first perform interpolation
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          buff_x[dx + 1] = x(ids, idc, idalpha, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

  // backpolate the error
  T buff_grad_y[4];
  backpolate_cubic<T>(buff_grad_y, grad_idy, buff_y, val, idy - idy_f + 1, 4);

  T buff_grad_x[4];
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[3])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          buff_x[dx + 1] = x(ids, idc, idalpha, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      backpolate_cubic<T>(buff_grad_x, grad_idx, buff_x, buff_grad_y[dy+1], idx - idx_f + 1, 4);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[4])
          tficg::CudaAtomicAdd(&grad_x(ids, idc, idalpha, c_idx_y, c_idx_x),
                                    buff_grad_x[dx + 1]);
      }
    }
  }
  return out;
}

template <typename T, tficg::interpolation_t I>
__global__ void metamorphosisWarpGradKernel(
    const typename Tensor5<T>::ConstTensor x,
    const typename Tensor4<T>::ConstTensor phi,
    const typename Tensor5<T>::ConstTensor grad_out,
    const int ids, const int idc,
    typename Tensor5<T>::Tensor grad_x, typename Tensor4<T>::Tensor grad_phi) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const int idr = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < x.dimensions()[4] && idy < x.dimensions()[3] &&
      idr < x.dimensions()[2]) {
    // first load the displacement
    const T dx = phi(ids, idy, idx, 0);
    const T dy = phi(ids, idy, idx, 1);
    const T beta = phi(ids, idy, idx, 2);

    // for each rototranslation space interpolate
    T grad_dx = 0;
    T grad_dy = 0;
    T grad_dr = 0;

    const int depth = grad_x.dimensions()[2];

    const int beta_offset = floorf(beta);
    const T d = beta - beta_offset;

    const int idalpha = idr + beta_offset;
    const int idalpha_f = mod(idalpha, depth);
    const int idalpha_c = mod(idalpha_f + 1, depth);

    // backpolate the gradient to the input and the deformation
    T grad_val = grad_out(ids, idc, idr, idy, idx);

    if (depth == 1)
    {
      switch (I) {
        case tficg::INTERPOLATE_BILINEAR:
        {
          const T val_f = backpolate_bilinear(grad_x, grad_dx, grad_dy, x, grad_val,
                                              ids, idc, idy + dy, idx + dx, 0);
          grad_dr = 0;
          break;
        }

        case tficg::INTERPOLATE_BICUBIC:
        {
          backpolate_bicubic(grad_x, grad_dx, grad_dy, x, grad_val,
                             ids, idc, idy + dy, idx + dx, 0);
          grad_dr = 0;
          break;
        }
      }
    }
    else
    {
      switch (I) {
        case tficg::INTERPOLATE_BILINEAR:
        {
          const T val_f = backpolate_bilinear(grad_x, grad_dx, grad_dy, x, grad_val * (1 - d),
                                      ids, idc, idy + dy, idx + dx, idalpha_f);
          const T val_c = backpolate_bilinear(grad_x, grad_dx, grad_dy, x, grad_val * d,
                                      ids, idc, idy + dy, idx + dx, idalpha_c);

          grad_dr = (val_c - val_f) * grad_val;
          break;
        }

        case tficg::INTERPOLATE_BICUBIC:
        {
          T buff[4];
          const int idalpha_f_1 = mod(idalpha - 1, depth);
          const int idalpha_c_1 = mod(idalpha + 2, depth);

          // first interpolate
          buff[0] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_f_1);
          buff[1] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_f);
          buff[2] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_c);
          buff[3] = interpolate_bicubic(x, ids, idc, idy + dy, idx + dx, idalpha_c_1);

          // distribute the gradient
          T buff_grad_alpha[4];
          backpolate_cubic<T>(buff_grad_alpha, grad_dr, buff, grad_val, d + 1, 4);

          backpolate_bicubic(grad_x, grad_dx, grad_dy, x, buff_grad_alpha[0],
                            ids, idc, idy + dy, idx + dx, idalpha_f_1);
          backpolate_bicubic(grad_x, grad_dx, grad_dy, x, buff_grad_alpha[1],
                            ids, idc, idy + dy, idx + dx, idalpha_f);
          backpolate_bicubic(grad_x, grad_dx, grad_dy, x, buff_grad_alpha[2],
                            ids, idc, idy + dy, idx + dx, idalpha_c);
          backpolate_bicubic(grad_x, grad_dx, grad_dy, x, buff_grad_alpha[3],
                            ids, idc, idy + dy, idx + dx, idalpha_c_1);
          break;
        }
      }
    }

    if (grad_dx != 0)
      tficg::CudaAtomicAdd(&grad_phi(ids, idy, idx, 0), grad_dx);
    if (grad_dy != 0)
      tficg::CudaAtomicAdd(&grad_phi(ids, idy, idx, 1), grad_dy);
    if (grad_dr != 0)
      tficg::CudaAtomicAdd(&grad_phi(ids, idy, idx, 2), grad_dr);
  }
}

template <typename T, tficg::interpolation_t I>
struct MetamorphosisInterpolationGradFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor5<T>::ConstTensor& x,
                  const typename Tensor4<T>::ConstTensor& phi,
                  const typename Tensor5<T>::ConstTensor& grad_out,
                  typename Tensor5<T>::Tensor& grad_x,
                  typename Tensor4<T>::Tensor& grad_phi) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // first clear the weight gradient
    tficg::fill<T, 4>(d, grad_phi, 0);
    tficg::fill<T, 5>(d, grad_x, 0);

    dim3 block_size(16, 16, 1);
    dim3 grid_size(iu::divUp(x.dimensions()[4], block_size.x),
                   iu::divUp(x.dimensions()[3], block_size.y), x.dimensions()[2]);

    for (int ids = 0; ids < x.dimensions()[0]; ++ids) {
      for (int idc = 0; idc < x.dimensions()[1]; ++idc)
        metamorphosisWarpGradKernel<T, I><<<grid_size, block_size, 0, d.stream()>>>(
            x, phi, grad_out, ids, idc, grad_x, grad_phi);
    }
  }
};

#define REGISTER_GPU_FUNCTOR(T)                          \
  template struct MetamorphosisInterpolationGradFunctor< \
      GPUDevice, T, tficg::INTERPOLATE_BILINEAR>;        \
  template struct MetamorphosisInterpolationGradFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


template <typename T>
__device__ T backpolate_bilinear(typename Tensor4<T>::Tensor& grad_x,
                                 T& grad_idx, T& grad_idy,
                                 const typename Tensor4<T>::ConstTensor& x,
                                 T val, int ids, int idc, T idy, T idx) {
  const int idx_f = floorf(idx);
  const int idy_f = floorf(idy);

  const int idx_c = idx_f + 1;
  const int idy_c = idy_f + 1;

  const T w = idx - idx_f;
  const T h = idy - idy_f;

  T i_ff = 0, i_fc = 0;
  if (idx_f >= 0 && idx_f < grad_x.dimensions()[3]) {
    if (idy_f >= 0 && idy_f < grad_x.dimensions()[2]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_f, idx_f),
                                (1 - h) * (1 - w) * val);
      i_ff = x(ids, idc, idy_f, idx_f);
    }

    if (idy_c >= 0 && idy_c < grad_x.dimensions()[2]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_c, idx_f),
                                h * (1 - w) * val);
      i_fc = x(ids, idc, idy_c, idx_f);
    }
  }

  T i_cf = 0, i_cc = 0;
  if (idx_c >= 0 && idx_c < grad_x.dimensions()[3]) {
    if (idy_f >= 0 && idy_f < grad_x.dimensions()[2]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_f, idx_c),
                                (1 - h) * w * val);
      i_cf = x(ids, idc, idy_f, idx_c);
    }

    if (idy_c >= 0 && idy_c < grad_x.dimensions()[2]) {
      tficg::CudaAtomicAdd(&grad_x(ids, idc, idy_c, idx_c),
                                h * w * val);
      i_cc = x(ids, idc, idy_c, idx_c);
    }
  }

  grad_idx += ((1 - h) * (i_cf - i_ff) + h * (i_cc - i_fc)) * val;
  grad_idy += ((1 - w) * (i_fc - i_ff) + w * (i_cc - i_cf)) * val;

  T out = (1 - h) * (1 - w) * i_ff;
  out += (1 - h) * w * i_cf;
  out += h * (1 - w) * i_fc;
  out += h * w * i_cc;

  return out;
}


template<typename T>
__device__ T backpolate_bicubic(typename Tensor4<T>::Tensor& grad_x,
  T& grad_idx, T& grad_idy,
  const typename Tensor4<T>::ConstTensor& x,
  T val, int ids, int idc, T idy, T idx)
{
  const int idy_f = floorf(idy);
  const int idx_f = floorf(idx);

  T buff_y[4];
  T buff_x[4];

  // first perform interpolation
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[2])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
          buff_x[dx + 1] = x(ids, idc, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      buff_y[dy + 1] = interpolate_cubic<T>(buff_x, idx - idx_f + 1, 4);
    }
    else
      buff_y[dy + 1] = 0;
  }

  T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

  // backpolate the error
  T buff_grad_y[4];
  backpolate_cubic<T>(buff_grad_y, grad_idy, buff_y, val, idy - idy_f + 1, 4);

  T buff_grad_x[4];
  for (int dy = -1; dy < 3; ++dy)
  {
    const int c_idx_y = idy_f + dy;

    if (c_idx_y >= 0 && c_idx_y < x.dimensions()[2])
    {
      // get the input values
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
          buff_x[dx + 1] = x(ids, idc, c_idx_y, c_idx_x);
        else
          buff_x[dx + 1] = 0;
      }
      backpolate_cubic<T>(buff_grad_x, grad_idx, buff_x, buff_grad_y[dy+1], idx - idx_f + 1, 4);
      for (int dx = -1; dx < 3; ++dx)
      {
        const int c_idx_x = idx_f + dx;
        if (c_idx_x >= 0 && c_idx_x < x.dimensions()[3])
          tficg::CudaAtomicAdd(&grad_x(ids, idc, c_idx_y, c_idx_x),
                                    buff_grad_x[dx + 1]);
      }
    }
  }
  return out;
}


template <typename T, tficg::interpolation_t I>
__global__ void warpGradKernel(
    const typename Tensor4<T>::ConstTensor x,
    const typename Tensor4<T>::ConstTensor phi,
    const typename Tensor4<T>::ConstTensor grad_out,
    const int ids, 
    typename Tensor4<T>::Tensor grad_x, typename Tensor4<T>::Tensor grad_phi) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int idy = threadIdx.y + blockIdx.y * blockDim.y;
  const int idc = threadIdx.z + blockIdx.z * blockDim.z;

  if (idx < x.dimensions()[3] && idy < x.dimensions()[2]) {
    // first load the displacement
    const T dx = phi(ids, idy, idx, 0);
    const T dy = phi(ids, idy, idx, 1);

    // for each rototranslation space interpolate
    T grad_dx = 0;
    T grad_dy = 0;

    // backpolate the gradient to the input and the deformation
    T grad_val = grad_out(ids, idc, idy, idx);

    switch (I) {
      case tficg::INTERPOLATE_BILINEAR:
      {
        const T val_f = backpolate_bilinear(grad_x, grad_dx, grad_dy, x, grad_val,
                                            ids, idc, idy + dy, idx + dx);
        break;
      }

      case tficg::INTERPOLATE_BICUBIC:
      {
        backpolate_bicubic(grad_x, grad_dx, grad_dy, x, grad_val,
                            ids, idc, idy + dy, idx + dx);
        break;
      }
    }

    if (grad_dx != 0)
      tficg::CudaAtomicAdd(&grad_phi(ids, idy, idx, 0), grad_dx);
    if (grad_dy != 0)
      tficg::CudaAtomicAdd(&grad_phi(ids, idy, idx, 1), grad_dy);
  }
}


template <typename T, tficg::interpolation_t I>
struct InterpolationGradFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor4<T>::ConstTensor& x,
                  const typename Tensor4<T>::ConstTensor& phi,
                  const typename Tensor4<T>::ConstTensor& grad_out,
                  typename Tensor4<T>::Tensor& grad_x,
                  typename Tensor4<T>::Tensor& grad_phi) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // first clear the weight gradient
    tficg::fill<T, 4>(d, grad_phi, 0);
    tficg::fill<T, 4>(d, grad_x, 0);

    dim3 block_size(8, 8, 1);
    dim3 grid_size(iu::divUp(x.dimensions()[3], block_size.x),
                   iu::divUp(x.dimensions()[2], block_size.y), x.dimensions()[1]);

    for (int ids = 0; ids < x.dimensions()[0]; ++ids)
        warpGradKernel<T, I><<<grid_size, block_size, 0, d.stream()>>>(
            x, phi, grad_out, ids, grad_x, grad_phi);
  }
};

#define REGISTER_GPU_FUNCTOR(T)                          \
  template struct InterpolationGradFunctor< \
      GPUDevice, T, tficg::INTERPOLATE_BILINEAR>;        \
  template struct InterpolationGradFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

#endif
