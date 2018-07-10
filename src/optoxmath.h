
#pragma once

#include <cuda.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>

#include <iu/iucore/lineardevicememory.h>

namespace optox{

namespace math{

template<typename T, unsigned int N, typename L>
void fill(iu::LinearDeviceMemory<T, N> &dst, const L &val, const cudaStream_t &stream)
{
    thrust::fill(thrust::cuda::par.on(stream), dst.begin(),dst.end(),val);
}

template<typename T, unsigned int N, typename L>
void mulC(iu::LinearDeviceMemory<T, N> &src, const L &val,
          iu::LinearDeviceMemory<T, N> &dst, const cudaStream_t &stream)
{
    thrust::transform(thrust::cuda::par.on(stream), src.begin(), src.end(),
        thrust::constant_iterator<T>(val),dst.begin(), thrust::multiplies<T>());
}

template <typename T>
struct weightedsum_transform_tuple :
        public thrust::unary_function< thrust::tuple<T, T>, T>
{
    typedef typename thrust::tuple<T,T> InputTuple;
    T w1,w2;
    weightedsum_transform_tuple(T _w1, T _w2) : w1(_w1), w2(_w2) {}
    __host__ __device__
    T operator()(const InputTuple& t) const
    {
        return thrust::get<0>(t)*w1+thrust::get<1>(t)*w2;
    }
};

template<typename T, unsigned int N>
void addWeighted(const iu::LinearDeviceMemory<T, N> &src1, const T &weight1,
                 const iu::LinearDeviceMemory<T, N> &src2, const T &weight2,
                 iu::LinearDeviceMemory<T, N> &dst,
                 const cudaStream_t &stream)
{
    weightedsum_transform_tuple<T> unary_op(weight1, weight2);
    thrust::transform(thrust::cuda::par.on(stream), 
                        thrust::make_zip_iterator(thrust::make_tuple(src1.begin(), src2.begin())),
                        thrust::make_zip_iterator(thrust::make_tuple(src1.end(), src2.end())),
                        dst.begin(),
                        unary_op);
}

}

}
