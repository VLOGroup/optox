#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/register_types.h"

#include "tf_filters.h"
#include "tf_cuutils.cuh"
#include "tf_activations.h"

// Rotate Filters Functor

/**
 * perform bilinear interpolation on the input image i_in given the
 * index (idx,idy)
 */
 template<typename T>
 __device__ T interpolate_bilinear(T *in, T idx, T idy, int kernel_width, int kernel_height)
 {
   const int idx_f = floorf(idx);
   const int idy_f = floorf(idy);

   const int idx_c = idx_f + 1;
   const int idy_c = idy_f + 1;

   const T w = idx - idx_f;
   const T h = idy - idy_f;

   T i_ff = 0, i_fc = 0;
   if (idx_f >= 0 && idx_f < kernel_width)
   {
      if (idy_f >= 0 && idy_f < kernel_height)
        i_ff = in[idx_f+kernel_width*idy_f];

      if (idy_c >= 0 && idy_c < kernel_height)
        i_fc = in[idx_f+kernel_width*idy_c];
   }

   T i_cf = 0, i_cc = 0;
   if (idx_c >= 0 && idx_c < kernel_width)
   {
      if (idy_f >= 0 && idy_f < kernel_height)
        i_cf = in[idx_c+kernel_width*idy_f];

      if (idy_c >= 0 && idy_c < kernel_height)
        i_cc = in[idx_c+kernel_width*idy_c];
   }

   T out = (1 - h) * (1 - w) * i_ff;
   out += (1 - h) * w * i_cf;
   out += h * (1 - w) * i_fc;
   out += h * w * i_cc;

   return out;
 }

 /**
  * perform bicubic interpolation on the input image i_in given the
  * index (idx,idy)
  */
 template<typename T>
 __device__ T interpolate_cubic(volatile T *in, T idx, int kernel_size)
 {
   const int idx_f = floorf(idx);
   const int idx_f_1 = idx_f - 1;
   const int idx_c = idx_f+1;
   const int idx_c_1 = idx_c+1;

   // get the input values
   T i_f = 0;
   if (idx_f >= 0 && idx_f < kernel_size)
     i_f = in[idx_f];
   T i_f_1 = 0;
   if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
     i_f_1 = in[idx_f_1];
   T i_c = 0;
   if (idx_c >= 0 && idx_c < kernel_size)
     i_c = in[idx_c];
   T i_c_1 = 0;
   if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
     i_c_1 = in[idx_c_1];

   // determine the coefficients
   const T p_f = i_f;
   const T p_prime_f = (i_c - i_f_1) / 2;
   const T p_c = i_c;
   const T p_prime_c = (i_c_1 - i_f) / 2;

   const T a = 2*p_f - 2*p_c + p_prime_f + p_prime_c;
   const T b = -3*p_f + 3*p_c - 2*p_prime_f - p_prime_c;
   const T c = p_prime_f;
   const T d = p_f;

   const T u = idx - idx_f;

   T out = u*(u*(u*a+b)+c) + d;

   return out;
 }

 template<typename T>
 __device__ T interpolate_bicubic(volatile T *in, T idx, T idy, int kernel_width, int kernel_height)
 {
   const int idy_f = floorf(idy);

   T buff_y[4];

   for (int dy = -1; dy < 3; ++dy)
   {
     const int c_idx_y = idy_f + dy;

     if (c_idx_y >= 0 && c_idx_y < kernel_height)
       buff_y[dy+1] = interpolate_cubic<T>(&(in[kernel_width*c_idx_y]), idx, kernel_width);
     else
       buff_y[dy+1] = 0;
   }

   T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

   return out;
 }

template<typename T, tficg::interpolation_t I>
__global__ void rotateFilterKernel(
    const typename Tensor4<T>::ConstTensor x,
    const typename Tensor1<T>::ConstTensor angles,
    typename Tensor5<T>::Tensor out)
{
  const int idx_out = threadIdx.x;
  const int idy_out = threadIdx.y;
  const int idz_in = threadIdx.z + blockIdx.z * blockDim.z;

  const int ids = idz_in % x.dimensions()[3];
  const int idf = idz_in / x.dimensions()[3];

  const int kernel_height = x.dimensions()[0];
  const int kernel_width = x.dimensions()[1];

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *x_shared = reinterpret_cast<T*>(s_buffer);

  // load data into shared memory (used to interpolate)
  x_shared[idx_out+kernel_width*idy_out] = x(idx_out, idy_out, idf, ids);

  __syncthreads();

  const int center_x = kernel_width / 2;
  const int center_y = kernel_height / 2;
  // compute the centralized coordinates
  const int idx_out_c = idx_out - center_x;
  const int idy_out_c = -idy_out + center_y;

  for (int angle_idx = 0; angle_idx < angles.dimensions()[0]; ++angle_idx)
  {
    const T theta = angles(angle_idx);
    // rotate the coordinate into the source image (-theta)
    const T cos_theta = __cosf(theta);
    const T sin_theta = __sinf(theta);
    const T idx_in_c = cos_theta * idx_out_c + sin_theta * idy_out_c;
    const T idy_in_c = -sin_theta * idx_out_c + cos_theta * idy_out_c;

    // transfer them back into the input image (in and out have same size)
    const T idx_in = idx_in_c + center_x;
    const T idy_in = -idy_in_c + center_y;

    switch(I)
    {
      case tficg::INTERPOLATE_BILINEAR:
        out(idx_out, idy_out, idf, ids, angle_idx) = interpolate_bilinear<T>(x_shared, idx_in,
          idy_in, kernel_width, kernel_height);
        break;
      case tficg::INTERPOLATE_BICUBIC:
        out(idx_out, idy_out, idf, ids, angle_idx) = interpolate_bicubic<T>(x_shared, idx_in,
          idy_in, kernel_width, kernel_height);
        break;
      case tficg::INTERPOLATE_INVALID:
        break;
    }
  }
}

template <typename T, tficg::interpolation_t I>
struct RotateFilterFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor4<T>::ConstTensor &x,
                  const typename Tensor1<T>::ConstTensor &angles,
                  typename Tensor5<T>::Tensor &out) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    dim3 block_size(x.dimensions()[0],
                    x.dimensions()[1]);
    dim3 grid_size(1, 1, x.dimensions()[2]*x.dimensions()[3]);

    unsigned int smem_size = x.dimensions()[0] * x.dimensions()[1] * sizeof(T);
    rotateFilterKernel<T,I><<<grid_size, block_size, smem_size, d.stream()>>>(
        x, angles, out);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct RotateFilterFunctor<GPUDevice, T, tficg::INTERPOLATE_BILINEAR>; \
    template struct RotateFilterFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


/**
 * perform bilinear interpolation on the input image i_in given the
 * index (idx,idy)
 */
 template<typename T>
 __device__ void backpolate_cubic(T *out, T error, T idx, int kernel_size, bool direct_buf = false)
 {
   const int idx_f = floor(idx);
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

   if (not direct_buf)
   {
     if (idx_f >= 0 && idx_f < kernel_size)
      tficg::CudaAtomicAdd(out + idx_f,   d_out_d_p_f   * error);
     if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
      tficg::CudaAtomicAdd(out + idx_f_1, d_out_d_p_f_1 * error);
     if (idx_c >= 0 && idx_c < kernel_size)
      tficg::CudaAtomicAdd(out + idx_c,   d_out_d_p_c   * error);
     if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
      tficg::CudaAtomicAdd(out + idx_c_1, d_out_d_p_c_1 * error);
   }
   else
   {
     if (idx_f >= 0 && idx_f < kernel_size)
       out[idx_f]   = d_out_d_p_f   * error;
     if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
       out[idx_f_1] = d_out_d_p_f_1 * error;
     if (idx_c >= 0 && idx_c < kernel_size)
       out[idx_c]   = d_out_d_p_c   * error;
     if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
       out[idx_c_1] = d_out_d_p_c_1 * error;
   }
 }

 template<typename T>
 __device__ void backpolate_bicubic(T *out, T error, T idx, T idy, int kernel_width, int kernel_height)
 {
   const int idy_f = floor(idy);

   T buff_y[4] = {0,0,0,0};
   backpolate_cubic<T>(buff_y, error, idy - idy_f + 1, 4, true);

   for (int dy = -1; dy < 3; ++dy)
   {
     const int c_idx_y = idy_f + dy;
     if (c_idx_y >= 0 && c_idx_y < kernel_height)
       backpolate_cubic<T>(&(out[kernel_width*c_idx_y]), buff_y[dy+1], idx, kernel_width);
   }
 }

 /**
  * perform bilinear interpolation adjoint on the output image i_out given the
  * value and the index (idx,idy)
  */
 template<typename T>
 __device__ void backpolate_bilinear(T *out, T val, T idx, T idy, int kernel_width, int kernel_height)
 {
   const int idx_f = floor(idx);
   const int idy_f = floor(idy);

   const int idx_c = idx_f + 1;
   const int idy_c = idy_f + 1;

   const T w = idx - idx_f;
   const T h = idy - idy_f;

   if (idx_f >= 0 && idx_f < kernel_width)
   {
      if (idy_f >= 0 && idy_f < kernel_height)
        tficg::CudaAtomicAdd(out + idx_f+kernel_width*idy_f, (1 - h) * (1 - w) * val);

      if (idy_c >= 0 && idy_c < kernel_height)
        tficg::CudaAtomicAdd(out + idx_f+kernel_width*idy_c, h * (1 - w) * val);
   }

   if (idx_c >= 0 && idx_c < kernel_width)
   {
      if (idy_f >= 0 && idy_f < kernel_height)
        tficg::CudaAtomicAdd(out + idx_c+kernel_width*idy_f, (1 - h) * w * val);

      if (idy_c >= 0 && idy_c < kernel_height)
        tficg::CudaAtomicAdd(out + idx_c+kernel_width*idy_c, h * w * val);
   }
 }


template<typename T, tficg::interpolation_t I>
__global__ void rotateFilterGradKernel(
    const typename Tensor1<T>::ConstTensor angles,
    const typename Tensor5<T>::ConstTensor grad_out,
    typename Tensor4<T>::Tensor grad_x)
{
  const int idx_in = threadIdx.x;
  const int idy_in = threadIdx.y;
  const int idz_out = threadIdx.z + blockIdx.z * blockDim.z;

  const int ids = idz_out % grad_x.dimensions()[3];
  const int idf = idz_out / grad_x.dimensions()[3];

  const int kernel_height = grad_x.dimensions()[0];
  const int kernel_width = grad_x.dimensions()[1];

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *grad_x_shared = reinterpret_cast<T*>(s_buffer);

  // load data into shared memory (used to interpolate)
  grad_x_shared[idx_in+kernel_width*idy_in] = 0;

  __syncthreads();

  const int center_x = kernel_width / 2;
  const int center_y = kernel_height / 2;
  // compute the centralized coordinates
  const int idx_in_c = idx_in - center_x;
  const int idy_in_c = -idy_in + center_y;

  for (int angle_idx = 0; angle_idx < angles.dimensions()[0]; ++angle_idx)
  {
    const T theta = angles(angle_idx);
    // rotate the coordinate into the source image (-theta)
    const T cos_theta = __cosf(theta);
    const T sin_theta = __sinf(theta);
    const T idx_out_c = cos_theta * idx_in_c + sin_theta * idy_in_c;
    const T idy_out_c = -sin_theta * idx_in_c + cos_theta * idy_in_c;

    // transfer them back into the input image (in and out have same size)
    const T idx_out = idx_out_c + center_x;
    const T idy_out = -idy_out_c + center_y;

    switch(I)
    {
      case tficg::INTERPOLATE_BILINEAR:
        backpolate_bilinear<T>(grad_x_shared, grad_out(idx_in, idy_in, idf, ids, angle_idx),
          idx_out, idy_out, kernel_width, kernel_height);
        break;
      case tficg::INTERPOLATE_BICUBIC:
        backpolate_bicubic<T>(grad_x_shared, grad_out(idx_in, idy_in, idf, ids, angle_idx),
          idx_out, idy_out, kernel_width, kernel_height);
        break;
      case tficg::INTERPOLATE_INVALID:
        break;
    }
  }

  __syncthreads();

  // write the result into global memory
  grad_x(idx_in, idy_in, idf, ids) = grad_x_shared[idx_in+kernel_width*idy_in];
}

template <typename T, tficg::interpolation_t I>
struct RotateFilterGradFunctor<GPUDevice, T, I> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor1<T>::ConstTensor &angles,
                  const typename Tensor5<T>::ConstTensor &grad_out,
                  typename Tensor4<T>::Tensor &grad_x) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // first clear the weight gradient
    tficg::fill<T, 4>(d, grad_x, 0);

    dim3 block_size(grad_x.dimensions()[0],
                    grad_x.dimensions()[1]);
    dim3 grid_size(1, 1, grad_x.dimensions()[2]*grad_x.dimensions()[3]);

    // unsigned int block_count = divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = grad_x.dimensions()[0] * grad_x.dimensions()[1] * sizeof(T);
    rotateFilterGradKernel<T,I><<<grid_size, block_size, smem_size, d.stream()>>>(
        angles, grad_out, grad_x);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct RotateFilterGradFunctor<GPUDevice, T, tficg::INTERPOLATE_BILINEAR>; \
    template struct RotateFilterGradFunctor<GPUDevice, T, tficg::INTERPOLATE_BICUBIC>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

#endif
