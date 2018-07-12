
#pragma once

#include <cuda.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <iu/iucore/lineardevicememory.h>

// #define OPTOX_MATH_DEBUG

namespace optox{

namespace math{

template<typename T, unsigned int N>
void fill(iu::LinearDeviceMemory<T, N> &dst, const T &val, const cudaStream_t &stream)
{
    thrust::fill(thrust::cuda::par.on(stream), dst.begin(),dst.end(),val);
#ifdef OPTOX_MATH_DEBUG
    IU_CUDA_CHECK;
#endif
}

template<typename T, unsigned int N>
void mulC(const iu::LinearDeviceMemory<T, N> &src, const T &val,
          iu::LinearDeviceMemory<T, N> &dst, const cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par.on(stream), src.begin(), src.end(),
        thrust::constant_iterator<T>(val),dst.begin(), thrust::multiplies<T>());
#ifdef OPTOX_MATH_DEBUG
    IU_CUDA_CHECK;
#endif
}

template <typename T>
struct WeightedSumBinaryFunction : public thrust::binary_function<const T, const T, T>
{
    T w_1, w_2;
    WeightedSumBinaryFunction(T _w1, T _w2): w_1(_w1), w_2(_w2) {}
    __host__ __device__
    T operator()(const T x, const T y) const
    { 
        return w_1 * x + w_2 * y;
    }
};

template<typename T, unsigned int N>
void addWeighted(const iu::LinearDeviceMemory<T, N> &src_1, const T &weight_1,
                 const iu::LinearDeviceMemory<T, N> &src_2, const T &weight_2,
                 iu::LinearDeviceMemory<T, N> &dst,
                 const cudaStream_t &stream)
{
    WeightedSumBinaryFunction<T> binary_op(weight_1, weight_2);
    thrust::transform(thrust::cuda::par.on(stream),
                        src_1.begin(), src_1.end(),
                        src_2.begin(),
                        dst.begin(),
                        binary_op);
#ifdef OPTOX_MATH_DEBUG
    IU_CUDA_CHECK;
#endif
}

}

}
