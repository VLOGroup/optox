///@file utils.cuh
///@brief Helper functions for parallel reduction
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

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
