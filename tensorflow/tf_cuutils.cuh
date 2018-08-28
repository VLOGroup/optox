
#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

using GPUDevice = Eigen::GpuDevice;


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;

  unsigned long long int old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

namespace tficg
{

unsigned int nextPowerof2(unsigned int v)
{
  v--;
  v |= v >> 2;
  v |= v >> 1;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

template <typename T, int NDIMS>
void fill(const GPUDevice &d,
          typename tensorflow::TTypes<T, NDIMS>::Tensor &x, T value)
{
  thrust::fill(thrust::cuda::par.on(d.stream()),
               thrust::device_ptr<T>(x.data()),
               thrust::device_ptr<T>(x.data() + x.size()),
               value);
}

template <typename T>
__device__ inline T CudaAtomicAdd(T *ptr, T value)
{
  return atomicAdd(ptr, value);
}

} // namespace tficg
