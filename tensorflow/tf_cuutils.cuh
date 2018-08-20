
#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

using GPUDevice = Eigen::GpuDevice;

namespace tficg {

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

template<typename T, int NDIMS>
void fill(const GPUDevice &d,
          typename tensorflow::TTypes<T,NDIMS>::Tensor &x, T value)
{
  thrust::fill(thrust::cuda::par.on(d.stream()),
                thrust::device_ptr<T>(x.data()),
                thrust::device_ptr<T>(x.data() + x.size()),
                value);
}

template <typename T>
__device__ inline T CudaAtomicAdd(T* ptr, T value) {
  return atomicAdd(ptr, value);
}

template <typename T, typename F>
__device__ T CudaAtomicCasHelper(T* ptr, F accumulate) {
  T old = *ptr;
  T assumed;
  do {
    assumed = old;
    old = atomicCAS(ptr, assumed, accumulate(assumed));
  } while (assumed != old);
  return old;
}

#if __CUDA_ARCH__ < 600
__device__ inline double CudaAtomicAdd(double* ptr, double value) {
  return CudaAtomicCasHelper(ptr, [value](double a) { return a + value; });
}
#endif
}
