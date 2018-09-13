
#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

using GPUDevice = Eigen::GpuDevice;

#define ICGVN_BLOCK_SIZE_3D_X 16
#define ICGVN_BLOCK_SIZE_3D_Y 8
#define ICGVN_BLOCK_SIZE_3D_Z 4

#define ICGVN_BLOCK_SIZE_2D_X 32
#define ICGVN_BLOCK_SIZE_2D_Y 32

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

inline int divUp(int length, int block_size)
{
  return (length + block_size - 1) / block_size;
}

} // namespace tficg
