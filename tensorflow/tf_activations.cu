///@file tf_activations.cu
///@brief TF CUDA wrappers for activation functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 17.08.2018

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/register_types.h"

#include "tf_activations.h"
#include "tf_cuutils.cuh"

#include <iu/iucore.h>


#define CUDART_PI_F 3.141592654f

// weight parallel reduction
template<typename T>
__device__ inline void reduceSharedGradWeights(volatile T *grad_w_shared,
    unsigned int tid, unsigned int BS, unsigned int num_weights)
{
  // reduce the weights gradient
  unsigned int current_size = BS / 2;
  unsigned int current_width = 2;
  while(current_size >= 2)
  {
    for (int j = 0; j < num_weights; j += current_width)
    {
      const unsigned int i_w = tid/current_size + j;
      const unsigned int i_x = tid%current_size;
      if (i_w < num_weights)
        grad_w_shared[i_w*BS + i_x] += grad_w_shared[i_w*BS + i_x + current_size];
    }

    __syncthreads();

    current_size /= 2;
    current_width *= 2;
  }
}

// Radial basis function Activation Functor
template<typename T, tficg::DerivativeOrder N>
__global__ void activationRBFKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor w,
    typename Tensor2<T>::Tensor out,
    T v_min, T delta_mu, T sigma_2, T scaling, int feature_stride)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int tid = threadIdx.x;

  const unsigned int num_weights = w.dimensions()[1];

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *w_shared = reinterpret_cast<T*>(s_buffer);
  T *mu_shared = w_shared + num_weights;

  // initialize the rbf centers
  if (tid < num_weights)
    mu_shared[tid] = v_min + tid*delta_mu;

  for (int idw = 0; idw < w.dimensions()[0]; ++idw)
  {
    // load the weights into shared memory
    if (tid < num_weights)
      w_shared[tid] = w(idw, tid);

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      // compute the activation
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_pos = x(idx, idc);
        T out_tmp = 0;
        for (int j = 0; j < num_weights; ++j)
        {
          // construct the rbf
          const T dist = x_pos - mu_shared[j];
          const T inner_deriv = - dist * sigma_2;
          const T exponent = (dist * dist * sigma_2)/2;
          const T p = (sizeof(T) == 4 ? expf(-exponent) : exp(-exponent)) * scaling;
          // determine the order of the gradient
          switch(N)
          {
            case tficg::DO_ZERO:
              out_tmp += w_shared[j] * p;
            break;
            case tficg::DO_FIRST:
              out_tmp += w_shared[j] * inner_deriv * p;
            break;
            case tficg::DO_SECOND:
              out_tmp += w_shared[j] * (-sigma_2 + inner_deriv*inner_deriv) * p;
            break;
            case tficg::DO_INT:
              T sigma = sqrt(sigma_2/2);
              out_tmp += w_shared[j] * erff(dist*sigma);
            break;
          }
        }
        out(idx, idc) = out_tmp;
      }
    }
    // sync to avoid filling of w_shared before every thread finished!
    __syncthreads();
  }
}

template <typename T, tficg::DerivativeOrder N>
struct ActivationRBFFunctor<GPUDevice, T, N> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // compute the mu spacing and standard deviation
    T delta_mu = (v_max - v_min) / (w.dimensions()[1] - 1);
    T sigma = (v_max - v_min) / ((w.dimensions()[1] - 1));
    T sigma_2 = pow(sigma, -2);
    T scaling = 1/sqrt(2*CUDART_PI_F*sigma*sigma);

    unsigned int thread_per_block = 1024;
    unsigned int block_count = iu::divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = 2*w.dimensions()[1] * sizeof(T);
    activationRBFKernel<T,N><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, w, out, v_min, delta_mu, sigma_2, scaling, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationRBFFunctor<GPUDevice, T, tficg::DO_ZERO>; \
    template struct ActivationRBFFunctor<GPUDevice, T, tficg::DO_FIRST>; \
    template struct ActivationRBFFunctor<GPUDevice, T, tficg::DO_SECOND>; \
    template struct ActivationRBFFunctor<GPUDevice, T, tficg::DO_INT>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

template<typename T, tficg::DerivativeOrder N>
__global__ void activationRBFGradWKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor grad_out,
    typename Tensor2<T>::Tensor grad_w,
    T v_min, T delta_mu, T sigma_2, T scaling, int feature_stride)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int BS = blockDim.x;
  const unsigned int tid = threadIdx.x;

  const unsigned int num_weights = grad_w.dimensions()[1];

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *mu_shared = reinterpret_cast<T*>(s_buffer);
  T *grad_w_shared = mu_shared + num_weights;

  // initialize the rbf centers
  if (tid < num_weights)
    mu_shared[tid] = v_min + tid*delta_mu;

  for (int idw = 0; idw < grad_w.dimensions()[0]; ++idw)
  {
    // initalize the gradients w.r.t. w
    for (int j = 0; j < num_weights; ++j)
      grad_w_shared[tid + j*BS] = 0;

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      if (idx < x.dimensions()[0])
      {
        const T x_pos = x(idx, idc);
        const T grad_out_pos = grad_out(idx, idc);
        for (int j = 0; j < num_weights; ++j)
        {
          // construct the rbf
          const T dist = x_pos - mu_shared[j];
          const T inner_deriv = - dist * sigma_2;
          const T exponent = (dist * dist * sigma_2)/2;
          const T p = (sizeof(T) == 4 ? expf(-exponent) : exp(-exponent)) * scaling;
          // determine the order of the gradient
          switch(N)
          {
            case tficg::DO_ZERO:
              grad_w_shared[tid + j*BS] += p * grad_out_pos;
            break;
            case tficg::DO_FIRST:
              grad_w_shared[tid + j*BS] += inner_deriv * p * grad_out_pos;
            break;
            case tficg::DO_SECOND:
              grad_w_shared[tid + j*BS] += p * (-sigma_2 + inner_deriv*inner_deriv) * grad_out_pos;
            break;
          }
        }
      }
    }

    __syncthreads();

    // reduce the weights gradient
    reduceSharedGradWeights(grad_w_shared, tid, BS, num_weights);

    // add to global gradient w
    if (tid < num_weights)
    {
      tficg::CudaAtomicAdd(grad_w.data() + idw*num_weights + tid,
        grad_w_shared[tid*BS] + grad_w_shared[tid*BS + 1]);
    }
  }
}

template <typename T, tficg::DerivativeOrder N>
struct ActivationRBFGradWFunctor<GPUDevice, T, N> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // compute the mu spacing and standard deviation
    T delta_mu = (v_max - v_min) / (grad_w.dimensions()[1] - 1);
    T sigma = (v_max - v_min) / ((grad_w.dimensions()[1] - 1));
    T sigma_2 = pow(sigma, -2);
    T scaling = 1/sqrt(2*CUDART_PI_F*sigma*sigma);

    // first clear the weight gradient
    tficg::fill<T, 2>(d, grad_w, 0);

    // compute block size dependent on num weights
    const unsigned int shared_memory_size = 48 * 1024;
    unsigned int thread_per_block = tficg::nextPowerof2(shared_memory_size / (sizeof(T) * grad_w.dimensions()[1]))/2;
    OP_REQUIRES(context, thread_per_block >= 64 && thread_per_block >= grad_w.dimensions()[1],
      tensorflow::errors::ResourceExhausted("Activation uses too much shared memory!"));

    unsigned int block_count = iu::divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = (2 + thread_per_block)* grad_w.dimensions()[1] * sizeof(T);
    activationRBFGradWKernel<T,N><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, grad_out, grad_w, v_min, delta_mu, sigma_2, scaling, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationRBFGradWFunctor<GPUDevice, T, tficg::DO_ZERO>; \
    template struct ActivationRBFGradWFunctor<GPUDevice, T, tficg::DO_FIRST>; \
    template struct ActivationRBFGradWFunctor<GPUDevice, T, tficg::DO_SECOND>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


// cubic b-spline Activation Functor
template<typename T>
inline __device__ T b_spline_cubic(T x)
{
  x = fabs(x);
  const T a = 2.0f - x;

  if (x < 1.0f) return 2.0f/3.0f - 0.5f*x*x*a;
  else if (x < 2.0f) return a*a*a / 6.0f;
  else return 0.0f;
}

// first derivative of cubic spline base function
template<typename T>
inline __device__ T b_spline_cubic_prime(T x)
{
  if (-2.0f < x && x <= -1.0f) return 0.5f*x*x + 2.0f*x + 2.0f;
  else if (-1.0f < x && x <= 0.0f) return -1.5f*x*x - 2.0f*x;
  else if ( 0.0f < x && x <= 1.0f) return  1.5f*x*x - 2.0f*x;
  else if ( 1.0f < x && x <  2.0f) return -0.5f*x*x + 2.0f*x - 2.0f;
  else return 0.0f;
}

// second derivative of cubic spline base function
template<typename T>
inline __device__ T b_spline_cubic_double_prime(T x)
{
  x = fabs(x);

  if (x < 1.0f) return 3.0f*x - 2.0f;
  else if (x < 2.0f) return 2.0f - x;
  else return 0.0f;
}

// b-spline Activation Functor
template<typename T>
inline __device__ T b_spline_linear(T x)
{
  x = fabs(x);

  if (x < 1.0f) return 1.f - x;
  else return 0.0f;
}

// first derivative of linear spline base function
template<typename T>
inline __device__ T b_spline_linear_prime(T x)
{
  if (-1.0f < x && x < 0.f) return 1.f;
  else if (0.f < x && x < 1.f) return -1.f;
  else return 0.f;
}

// b-spline quadratic Activation Functor
template<typename T>
inline __device__ T b_spline_quad(T x)
{
  x = fabs(x);

  if (x <= 0.5f) return 0.75f - x*x;
  else if (x <= 1.5f) return (1.5f - x)*(1.5f - x)*0.5f;
  else return 0.f;
}

// first derivative of quadratic spline base function
template<typename T>
inline __device__ T b_spline_quad_prime(T x)
{
  if (-1.5f <= x && x < -0.5f) return x + 1.5f;
  else if (-0.5f <= x && x <= 0.5f) return -2*x;
  else if (0.5f <= x && x <= 1.5f) return x - 1.5f;
  else return 0.f;
}

template<typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
__global__ void activationBSplineKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor w,
    typename Tensor2<T>::Tensor out,
    T v_min, T v_max, int feature_stride)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int tid = threadIdx.x;

  const unsigned int num_weights = w.dimensions()[1];

  const T delta_1 = (num_weights - 1) / (v_max - v_min);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < w.dimensions()[0]; ++idw)
  {
    // load the weights into shared memory
    if (tid < num_weights)
      w_shared[tid] = w(idw, tid);

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      // compute the activation
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) * delta_1;
        if (S == tficg::SO_LINEAR && (x_idx <= 0 || x_idx >= num_weights - 1))
        {
          // etrapolation
          if (x_idx <= 0)
          {
            switch(N)
            {
              case tficg::DO_ZERO:
              out(idx, idc) = (x_idx - 0) * (w_shared[1] - w_shared[0]) + w_shared[0];
              break;
              case tficg::DO_FIRST:
              out(idx, idc) = (w_shared[1] - w_shared[0])*delta_1;
              break;
            }
          }
          if (x_idx >= num_weights - 1)
          {
            switch(N)
            {
              case tficg::DO_ZERO:
              out(idx, idc) = (x_idx - (num_weights-1)) * (w_shared[num_weights-1] - w_shared[num_weights-2]) + w_shared[num_weights-1];
              break;
              case tficg::DO_FIRST:
              out(idx, idc) = (w_shared[num_weights-1] - w_shared[num_weights-2])*delta_1;
              break;
            }
          }  
        }
        else if (x_idx < -2 || x_idx > num_weights + 1)
          out(idx, idc) = 0;
        else
        {
          // get the correct weight values
          const int x_idx_f = floor(x_idx);
          const T alpha = x_idx - x_idx_f;

          T out_tmp = 0;
          for (int dx = -1; dx <= 2; ++dx)
          {
            // first get the corresponding base function
            T b_spline_x = 0;
            switch(N)
            {
              case tficg::DO_ZERO:
                switch(S)
                {
                  case tficg::SO_LINEAR:
                  b_spline_x = b_spline_linear<T>(dx-alpha);
                  break;
                  case tficg::SO_QUADRATIC:
                  b_spline_x = b_spline_quad<T>(dx-alpha);
                  break;
                  case tficg::SO_CUBIC:
                  b_spline_x = b_spline_cubic<T>(dx-alpha);
                  break;
                }
              break;
              case tficg::DO_FIRST:
                switch(S)
                {
                  case tficg::SO_LINEAR:
                  b_spline_x = b_spline_linear_prime<T>(dx-alpha) * (-delta_1);
                  break;
                  case tficg::SO_QUADRATIC:
                  b_spline_x = b_spline_quad_prime<T>(dx-alpha) * (-delta_1);
                  break;
                  case tficg::SO_CUBIC:
                  b_spline_x = b_spline_cubic_prime<T>(dx-alpha) * (-delta_1);
                  break;
                }
              break;
              case tficg::DO_SECOND:
                b_spline_x = b_spline_cubic_double_prime<T>(dx-alpha) * (delta_1*delta_1);
              break;
            }
            // compute the current index
            const int idx_tmp = x_idx_f+dx;
            T w_tmp = 0;
            if (idx_tmp >= 0 && idx_tmp < num_weights)
              w_tmp = w_shared[idx_tmp];
            // add the fraction to the output
            out_tmp += b_spline_x * w_tmp;
          }
          out(idx, idc) = out_tmp;
        }
      }
    }
    // sync to avoid filling of w_shared before every thread finished!
    __syncthreads();
  }
}

template <typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineFunctor<GPUDevice, T, S, N> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    unsigned int thread_per_block = 1024;
    unsigned int block_count = iu::divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = w.dimensions()[1] * sizeof(T);
    activationBSplineKernel<T,S,N><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, w, out, v_min, v_max, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_FIRST>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_QUADRATIC, tficg::DO_ZERO>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_QUADRATIC, tficg::DO_FIRST>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>; \
    template struct ActivationBSplineFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_SECOND>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR

template<typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
__global__ void activationBSplineGradWKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor grad_out,
    typename Tensor2<T>::Tensor grad_w,
    T v_min, T v_max, int feature_stride)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int BS = blockDim.x;
  const unsigned int tid = threadIdx.x;

  const unsigned int num_weights = grad_w.dimensions()[1];

  const T delta_1 = (num_weights - 1) / (v_max - v_min);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *grad_w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < grad_w.dimensions()[0]; ++idw)
  {
    // initalize the gradients w.r.t. w
    for (int j = 0; j < num_weights; ++j)
      grad_w_shared[tid + j*BS] = 0;

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      if (idx < x.dimensions()[0])
      {
        const T x_pos = x(idx, idc);
        const T grad_out_pos = grad_out(idx, idc);

        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) * delta_1;

        if (S == tficg::SO_LINEAR && (x_idx <= 0 || x_idx >= num_weights - 1))
        {
          // etrapolation
          if (x_idx <= 0)
          {
            grad_w_shared[tid + 0*BS] += (-x_idx + 1) * grad_out_pos;
            grad_w_shared[tid + 1*BS] += x_idx * grad_out_pos;
          }
          if (x_idx >= num_weights - 1)
          {
            grad_w_shared[tid + (num_weights-2)*BS] += -(x_idx - (num_weights-1)) * grad_out_pos;
            grad_w_shared[tid + (num_weights-1)*BS] += ((x_idx - (num_weights-1)) + 1) * grad_out_pos;
          }  
        }
        else if (x_idx >= -2 && x_idx < num_weights + 1)
        {
          // get the correct weight values
          const int x_idx_f = floor(x_idx);
          const T alpha = x_idx - x_idx_f;

          for (int dx = -1; dx <= 2; ++dx)
          {
            // first get the corresponding base function
            T b_spline_x = 0;
            switch(N)
            {
              case tficg::DO_ZERO:
                switch(S)
                {
                  case tficg::SO_LINEAR:
                  b_spline_x = b_spline_linear<T>(dx-alpha);
                  break;
                  case tficg::SO_QUADRATIC:
                  b_spline_x = b_spline_quad<T>(dx-alpha);
                  break;
                  case tficg::SO_CUBIC:
                  b_spline_x = b_spline_cubic<T>(dx-alpha);
                  break;
                }
              break;
              case tficg::DO_FIRST:
                b_spline_x = b_spline_cubic_prime<T>(dx-alpha) * (-delta_1);
              break;
              case tficg::DO_SECOND:
                b_spline_x = b_spline_cubic_double_prime<T>(dx-alpha) * (delta_1*delta_1);
              break;
            }
            // compute the current index
            const int idx_tmp = x_idx_f+dx;
            if (idx_tmp >= 0 && idx_tmp < num_weights)
              grad_w_shared[tid + idx_tmp*BS] += b_spline_x * grad_out_pos;
          }
        }
      }
    }

    __syncthreads();

    // reduce the weights gradient
    reduceSharedGradWeights(grad_w_shared, tid, BS, num_weights);

    // add to global gradient w
    if (tid < num_weights)
    {
      T grad_w_tid = grad_w_shared[tid*BS] + grad_w_shared[tid*BS + 1];
      if (grad_w_tid != 0)
        tficg::CudaAtomicAdd(grad_w.data() + idw*num_weights + tid, grad_w_tid);
    }
  }
}

template <typename T, tficg::SplineOrder S, tficg::DerivativeOrder N>
struct ActivationBSplineGradWFunctor<GPUDevice, T, S, N> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // first clear the weight gradient
    tficg::fill<T, 2>(d, grad_w, 0);

    // compute block size dependent on num weights
    const unsigned int shared_memory_size = 48 * 1024;
    unsigned int thread_per_block = tficg::nextPowerof2(shared_memory_size / (sizeof(T) * grad_w.dimensions()[1]))/2;
    OP_REQUIRES(context, thread_per_block >= 64 && thread_per_block >= grad_w.dimensions()[1],
      tensorflow::errors::ResourceExhausted("Activation uses too much shared memory!"));

    unsigned int block_count = iu::divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = thread_per_block * grad_w.dimensions()[1] * sizeof(T);
    activationBSplineGradWKernel<T,S,N><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, grad_out, grad_w, v_min, v_max, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationBSplineGradWFunctor<GPUDevice, T, tficg::SO_LINEAR, tficg::DO_ZERO>; \
    template struct ActivationBSplineGradWFunctor<GPUDevice, T, tficg::SO_QUADRATIC, tficg::DO_ZERO>; \
    template struct ActivationBSplineGradWFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_ZERO>; \
    template struct ActivationBSplineGradWFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_FIRST>; \
    template struct ActivationBSplineGradWFunctor<GPUDevice, T, tficg::SO_CUBIC, tficg::DO_SECOND>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


// Linear interpolation Activation Functor
template<typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
__global__ void activationInterpolateLinearKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor w,
    typename Tensor2<T>::Tensor out,
    T v_min, T v_max, int feature_stride)
{
  const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const unsigned int tid = threadIdx.x;

  const unsigned int num_weights = w.dimensions()[1];

  const T delta_1 = (num_weights - 1) / (v_max - v_min);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < w.dimensions()[0]; ++idw)
  {
    // load the weights into shared memory
    if (tid < num_weights)
      w_shared[tid] = w(idw, tid);
    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      // compute the activation
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) * delta_1;
        // get the correct weight values
        const int x_idx_f = floor(x_idx);
        const int x_idx_c = x_idx_f + 1;
        T w_f = 0;
        T w_c = 0;
        T alpha = 0;
        if (x_idx_f >= 0 && x_idx_f < num_weights - 1)
        {
          w_f = w_shared[x_idx_f];
          w_c = w_shared[x_idx_c];
          alpha = x_idx - x_idx_f;
        }
        else if (x_idx_f < 0)
        {
          switch(TBorderMode)
          {
            case tficg::DO_NONE:
              if (x_idx >= -1)
              {
                w_c = w_shared[0];
                alpha = x_idx - x_idx_f;
              }
              break;
            case tficg::DO_EXTRAPOLATE:
              // extrapolation to the left
              w_f = w_shared[0];
              w_c = w_shared[1];
              alpha = x_idx;
              break;
          }
        }
        else if (x_idx_f >= num_weights - 1)
        {
          switch(TBorderMode)
          {
            case tficg::DO_NONE:
              if (x_idx < num_weights)
              {
                w_f = w_shared[num_weights-1];
                alpha = x_idx - x_idx_f;
              }
              break;
            case tficg::DO_EXTRAPOLATE:
              // extrapolation to the right
              w_f = w_shared[num_weights-2];
              w_c = w_shared[num_weights-1];
              alpha = x_idx - (num_weights-2);
              break;
          }
        }

        // determine the order of the gradient
        switch(N)
        {
          case tficg::DO_ZERO:
            out(idx, idc) = (1 - alpha) * w_f + alpha * w_c;
          break;
          case tficg::DO_FIRST:
            out(idx, idc) = (w_c - w_f) * delta_1;
          break;
        }
      }
    }
    // sync to avoid filling of w_shared before every thread finished!
    __syncthreads();
  }
}

// Linear interpolation Activation Functor
template<typename T, tficg::BorderMode TBorderMode>
__global__ void activationIntegralInterpolateLinearKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor w,
    typename Tensor2<T>::Tensor out,
    T v_min, T v_max, int feature_stride)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const int tid = threadIdx.x;

  const int num_weights = w.dimensions()[1];
  const int b = num_weights / 2;

  const T delta = (v_max - v_min) / (num_weights - 1);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < w.dimensions()[0]; ++idw)
  {
    // load the weights into shared memory
    if (tid < num_weights)
      w_shared[tid] = w(idw, tid);
    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      // compute the activation
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) / delta;
        // get the correct weight values
        const int x_idx_f = floorf(x_idx);
        const int x_idx_c = x_idx_f + 1;

        if (x_idx_f >= b)
        {
          if (x_idx_f < num_weights - 1)
          {
            T w_int = 0;
            for (int j = b; j < x_idx_f; ++j)
              w_int += w_shared[j+1] + w_shared[j];

            const T alpha = x_idx - x_idx_f;
            w_int += alpha * (alpha*w_shared[x_idx_c] + (2-alpha)*w_shared[x_idx_f]);
            out(idx, idc) = (w_int * delta) / 2;
          }
          else if (TBorderMode == tficg::DO_EXTRAPOLATE)
          {
            T w_int = 0;
            for (int j = b; j < num_weights-2; ++j)
              w_int += w_shared[j+1] + w_shared[j];

            const T alpha = x_idx - (num_weights-2);
            w_int += alpha * (alpha*w_shared[num_weights-1] + (2-alpha)*w_shared[num_weights-2]);
            out(idx, idc) = (w_int * delta) / 2;
          }
          else if (TBorderMode == tficg::DO_NONE)
          {
            T w_int = 0;
            for (int j = b; j < num_weights-2; ++j)
              w_int += w_shared[j+1] + w_shared[j];

            if (x_idx_f < num_weights)
            {
              const T alpha = x_idx - x_idx_f;
              w_int += w_shared[num_weights-1] + alpha * ((2-alpha)*w_shared[num_weights-1]);
            }
            else
            {
              w_int += 2*w_shared[num_weights-1];
            }

            out(idx, idc) = (w_int * delta) / 2;
          }
        }
        else 
        {
          if (x_idx_f >= 0)
          {
            T w_int = 0;
            for (int j = b; j > x_idx_c; --j)
              w_int += w_shared[j-1] + w_shared[j];

            const T alpha = x_idx - x_idx_f;
            w_int += (1 - alpha) * ((1+alpha)*w_shared[x_idx_c] + (1-alpha)*w_shared[x_idx_f]);
            out(idx, idc) = (w_int * -delta) / 2;
          }
          else if (TBorderMode == tficg::DO_EXTRAPOLATE)
          {
            T w_int = 0;
            for (int j = b; j > 0; --j)
                w_int += w_shared[j-1] + w_shared[j];

            const T alpha = x_idx;
            w_int += -alpha * (alpha*w_shared[1] + (2-alpha)*w_shared[0]);
            out(idx, idc) = (w_int * -delta) / 2;
          }
          else if (TBorderMode == tficg::DO_NONE)
          {
            T w_int = 0;
            for (int j = b; j > 0; --j)
              w_int += w_shared[j-1] + w_shared[j];

            if (x_idx_f >= -1)
            {
              const T alpha = x_idx - x_idx_f;
              w_int += (1 - alpha) * ((1+alpha)*w_shared[0]);
            }
            else
            {
              w_int += w_shared[0];
            }

            out(idx, idc) = (w_int * -delta) / 2;          
          }
        }
      }
    }
    // sync to avoid filling of w_shared before every thread finished!
    __syncthreads();
  }
}

template <typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
struct ActivationInterpolateLinearFunctor<GPUDevice, T, N, TBorderMode> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &w,
                  typename Tensor2<T>::Tensor &out,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    unsigned int thread_per_block = 1024;
    unsigned int block_count = iu::divUp(out.dimensions()[0], thread_per_block);
    unsigned int smem_size = w.dimensions()[1] * sizeof(T);
    if (N == tficg::DO_INT)
      activationIntegralInterpolateLinearKernel<T, TBorderMode><<<block_count, thread_per_block, smem_size, d.stream()>>>(
          x, w, out, v_min, v_max, feature_stride);
    else
      activationInterpolateLinearKernel<T, N, TBorderMode><<<block_count, thread_per_block, smem_size, d.stream()>>>(
          x, w, out, v_min, v_max, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_ZERO, tficg::DO_NONE>; \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_ZERO, tficg::DO_EXTRAPOLATE>; \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_FIRST, tficg::DO_NONE>; \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_FIRST, tficg::DO_EXTRAPOLATE>; \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_INT, tficg::DO_NONE>; \
    template struct ActivationInterpolateLinearFunctor<GPUDevice, T, tficg::DO_INT, tficg::DO_EXTRAPOLATE>; 
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR


// Gradient Functor
template<typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
__global__ void activationInterpolateLinearGradWKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor grad_out,
    typename Tensor2<T>::Tensor grad_w,
    T v_min, T v_max, int feature_stride)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const int BS = blockDim.x;
  const int tid = threadIdx.x;

  const int num_weights = grad_w.dimensions()[1];

  const T delta_1 = (num_weights - 1) / (v_max - v_min);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *grad_w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < grad_w.dimensions()[0]; ++idw)
  {
    // initalize the gradients w.r.t. w
    for (int j = 0; j < num_weights; ++j)
      grad_w_shared[tid + j*BS] = 0;

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) * delta_1;
        // get the correct weight values
        const int x_idx_f = floor(x_idx);
        const int x_idx_c = x_idx_f + 1;

        if (x_idx_f >= 0 && x_idx_f < num_weights - 1)
        {
          const T alpha = x_idx - x_idx_f;
          grad_w_shared[tid + x_idx_f*BS] += grad_out(idx, idc) * (1 - alpha);
          grad_w_shared[tid + x_idx_c*BS] += grad_out(idx, idc) * alpha;
        }
        else if (x_idx_f < 0 && TBorderMode == tficg::DO_EXTRAPOLATE)
        {
          // extrapolation to the left
          const T alpha = x_idx;
          grad_w_shared[tid + 0*BS] += grad_out(idx, idc) * (1 - alpha);
          grad_w_shared[tid + 1*BS] += grad_out(idx, idc) * alpha;
        }
        else if (x_idx_f >= num_weights - 1 && TBorderMode == tficg::DO_EXTRAPOLATE)
        {
          // extrapolation to the right
          const T alpha = x_idx - (num_weights-2);
          grad_w_shared[tid + (num_weights-2)*BS] += grad_out(idx, idc) * (1 - alpha);
          grad_w_shared[tid + (num_weights-1)*BS] += grad_out(idx, idc) * alpha;
        }
        else if (x_idx_f >= -1 && x_idx_f < 0 && TBorderMode == tficg::DO_NONE)
        {
          const T alpha = x_idx - x_idx_f;
          grad_w_shared[tid + 0*BS] += grad_out(idx, idc) * alpha;
        }
        else if (x_idx_f < num_weights && x_idx_f >= num_weights - 1 && TBorderMode == tficg::DO_NONE)
        {
          const T alpha = x_idx - x_idx_f;
          grad_w_shared[tid + (num_weights-1)*BS] += grad_out(idx, idc) * (1 - alpha);
        }
      }
    }

    __syncthreads();

    // reduce the weights gradient
    reduceSharedGradWeights(grad_w_shared, tid, BS, num_weights);

    // add to global gradient w
    if (tid < num_weights)
    {
      T grad_w_tid = grad_w_shared[tid*BS] + grad_w_shared[tid*BS + 1];
      if (grad_w_tid != 0)
        tficg::CudaAtomicAdd(grad_w.data() + idw*num_weights + tid, grad_w_tid);
    }
  }
}

template<typename T, tficg::BorderMode TBorderMode>
__global__ void activationIntegralInterpolateLinearGradWKernel(
    const typename Tensor2<T>::ConstTensor x,
    const typename Tensor2<T>::ConstTensor grad_out,
    typename Tensor2<T>::Tensor grad_w,
    T v_min, T v_max, int feature_stride)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  const int BS = blockDim.x;
  const int tid = threadIdx.x;

  const int num_weights = grad_w.dimensions()[1];
  const int b = num_weights / 2;

  const T delta = (v_max - v_min) / (num_weights - 1);

  extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
  T *grad_w_shared = reinterpret_cast<T*>(s_buffer);

  for (int idw = 0; idw < grad_w.dimensions()[0]; ++idw)
  {
    // initalize the gradients w.r.t. w
    for (int j = 0; j < num_weights; ++j)
      grad_w_shared[tid + j*BS] = 0;

    __syncthreads();

    for (int idc = idw*feature_stride; idc < (idw+1)*feature_stride; ++idc)
    {
      if (idx < x.dimensions()[0])
      {
        // compute the linear index
        const T x_idx = (x(idx, idc) - v_min) / delta;
        // get the correct weight values
        const int x_idx_f = floorf(x_idx);
        const int x_idx_c = x_idx_f + 1;

        const T grad_out_pos = (grad_out(idx, idc) * delta) / 2;

        if (x_idx_f >= b)
        {
          if (x_idx_f < num_weights - 1)
          {
            for (int j = b; j < x_idx_f; ++j)
            {
              grad_w_shared[tid + j*BS] += grad_out_pos;
              grad_w_shared[tid + (j+1)*BS] += grad_out_pos;
            }

            const T alpha = x_idx - x_idx_f;
            grad_w_shared[tid + x_idx_c*BS] += grad_out_pos * alpha * alpha;
            grad_w_shared[tid + x_idx_f*BS] += grad_out_pos * alpha * (2-alpha);
          }
          else if (TBorderMode == tficg::DO_EXTRAPOLATE)
          {
            for (int j = b; j < num_weights-2; ++j)
            {
              grad_w_shared[tid + j*BS] += grad_out_pos;
              grad_w_shared[tid + (j+1)*BS] += grad_out_pos;
            }

            const T alpha = x_idx - (num_weights-2);
            grad_w_shared[tid + (num_weights-1)*BS] += grad_out_pos * alpha * alpha;
            grad_w_shared[tid + (num_weights-2)*BS] += grad_out_pos * alpha * (2-alpha);
          }
          else if (TBorderMode == tficg::DO_NONE)
          {
            for (int j = b; j < num_weights-2; ++j)
            {
              grad_w_shared[tid + j*BS] += grad_out_pos;
              grad_w_shared[tid + (j+1)*BS] += grad_out_pos;
            }

            grad_w_shared[tid + (num_weights-1)*BS] += grad_out_pos;

            if (x_idx_f < num_weights)
            {
              const T alpha = x_idx - x_idx_f;
              grad_w_shared[tid + (num_weights-1)*BS] += grad_out_pos * alpha * (2-alpha);
            }
            else
            {
              grad_w_shared[tid + (num_weights-1)*BS] += grad_out_pos;
            }
          }
        }
        else 
        {
          if (x_idx_f >= 0)
          {
            for (int j = b; j > x_idx_c; --j)
            {
              grad_w_shared[tid + j*BS] -= grad_out_pos;
              grad_w_shared[tid + (j-1)*BS] -= grad_out_pos;
            }

            const T alpha = x_idx - x_idx_f;
            grad_w_shared[tid + x_idx_c*BS] -= grad_out_pos * (1-alpha) * (1+alpha);
            grad_w_shared[tid + x_idx_f*BS] -= grad_out_pos * (1-alpha) * (1-alpha);
          }
          else if (TBorderMode == tficg::DO_EXTRAPOLATE)
          {
            for (int j = b; j > 0; --j)
            {
              grad_w_shared[tid + j*BS] -= grad_out_pos;
              grad_w_shared[tid + (j-1)*BS] -= grad_out_pos;
            }

            const T alpha = x_idx;
            grad_w_shared[tid + 1*BS] -= grad_out_pos * (-alpha) * alpha;
            grad_w_shared[tid + 0*BS] -= grad_out_pos * (-alpha) * (2-alpha);
          }
          else if (TBorderMode == tficg::DO_NONE)
          {
            for (int j = b; j > 0; --j)
            {
              grad_w_shared[tid + j*BS] -= grad_out_pos;
              grad_w_shared[tid + (j-1)*BS] -= grad_out_pos;
            }

            if (x_idx_f >= -1)
            {
              const T alpha = x_idx - x_idx_f;
              grad_w_shared[tid + 0*BS] -= grad_out_pos * (1-alpha) * (1+alpha);
            }
            else
            {
              grad_w_shared[tid + 0*BS] -= grad_out_pos;
            }
          }
        }
      }
    }

    __syncthreads();

    // reduce the weights gradient
    reduceSharedGradWeights(grad_w_shared, tid, BS, num_weights);

    // add to global gradient w
    if (tid < num_weights)
    {
      T grad_w_tid = grad_w_shared[tid*BS] + grad_w_shared[tid*BS + 1];
      if (grad_w_tid != 0)
        tficg::CudaAtomicAdd(grad_w.data() + idw*num_weights + tid, grad_w_tid);
    }
  }
}

template <typename T, tficg::DerivativeOrder N, tficg::BorderMode TBorderMode>
struct ActivationInterpolateLinearGradWFunctor<GPUDevice, T, N, TBorderMode> {
  void operator()(tensorflow::OpKernelContext* context,
                  const typename Tensor2<T>::ConstTensor &x,
                  const typename Tensor2<T>::ConstTensor &grad_out,
                  typename Tensor2<T>::Tensor &grad_w,
                  T v_min, T v_max, int feature_stride) {
    const GPUDevice d = context->eigen_device<GPUDevice>();

    // first clear the weight gradient
    tficg::fill<T, 2>(d, grad_w, 0);

    // compute block size dependent on num weights
    const unsigned int shared_memory_size = 48 * 1024;
    unsigned int thread_per_block = tficg::nextPowerof2(shared_memory_size / (sizeof(T) * grad_w.dimensions()[1]))/2;
    OP_REQUIRES(context, thread_per_block >= 64 && thread_per_block >= grad_w.dimensions()[1],
      tensorflow::errors::ResourceExhausted("Activation uses too much shared memory!"));

    unsigned int block_count = iu::divUp(x.dimensions()[0], thread_per_block);
    unsigned int smem_size = thread_per_block* grad_w.dimensions()[1] * sizeof(T);

    if (N == tficg::DO_INT)
      activationIntegralInterpolateLinearGradWKernel<T, TBorderMode><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, grad_out, grad_w, v_min, v_max, feature_stride);
    else
      activationInterpolateLinearGradWKernel<T, N, TBorderMode><<<block_count, thread_per_block, smem_size, d.stream()>>>(
        x, grad_out, grad_w, v_min, v_max, feature_stride);
  }
};

#define REGISTER_GPU_FUNCTOR(T) \
    template struct ActivationInterpolateLinearGradWFunctor<GPUDevice, T, tficg::DO_ZERO, tficg::DO_NONE>; \
    template struct ActivationInterpolateLinearGradWFunctor<GPUDevice, T, tficg::DO_ZERO, tficg::DO_EXTRAPOLATE>; \
    template struct ActivationInterpolateLinearGradWFunctor<GPUDevice, T, tficg::DO_INT, tficg::DO_NONE>; \
    template struct ActivationInterpolateLinearGradWFunctor<GPUDevice, T, tficg::DO_INT, tficg::DO_EXTRAPOLATE>;
TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR
