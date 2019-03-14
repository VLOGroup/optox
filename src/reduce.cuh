///@file utils.cuh
///@brief Helper functions for parallel reduction
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#pragma once

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template<typename T>
__device__ __forceinline__ void warpReduce(volatile T* sdata, int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

template<typename T>
__device__ inline void parallelReduce(volatile T* sdata, int tid, int block_size)
{
    for(int s = block_size / 2; s > 32; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if(tid < 32)
        warpReduce(sdata, tid);
}
