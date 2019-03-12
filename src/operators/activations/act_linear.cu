///@file act_linear.cu
///@brief linear interpolation activation function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "act_linear.h"
#include "utils.cuh"


template<typename T>
__global__ void act_linear_forward_kernel(
    typename optox::DTensor<T, 2>::Ref output,
    const typename optox::DTensor<T, 2>::ConstRef input,
    const typename optox::DTensor<T, 2>::ConstRef weights,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.size_[0] || y >= input.size_[1])
        return;

    const int Nw = weights.size_[0];

    const T k = ((vmax - vmin) / (Nw - 1));

    const T inp_pos = input(x, y);
    const T idx = (inp_pos - vmin) / k;
    T val = 0;
    bool boundary = false;
    for (int i = 0; i < Nw; ++i)
    {
        // perform extrapolation
        if (i == 0 && idx <= i)
        {
            val = weights(i, y);
            boundary = true;
        }
        else if (i == Nw -1 && idx >= i)
        {
            val = weights(i, y);
            boundary = true;
        }
        else if (!boundary && idx >= i && idx < i + 1)
        {
            const T alpha = idx - i;
            val += weights(i, y) * (1 - alpha);
        }
        else if (!boundary && idx >= i - 1 && idx < i)
        {
            const T alpha = idx - i + 1;
            val += weights(i, y) * alpha;
        }
    }
    output(x, y) = val;
}


template<typename T>
__global__ void act_linear_backward_kernel(
    typename optox::DTensor<T, 2>::Ref grad_input,
    typename optox::DTensor<T, 2>::Ref grad_weights,
    const typename optox::DTensor<T, 2>::ConstRef input,
    const typename optox::DTensor<T, 2>::ConstRef weights,
    const typename optox::DTensor<T, 2>::ConstRef grad_output,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char sbuffer[];
    T *sdata = reinterpret_cast<T*>(sbuffer);

    if (x >= input.size_[0] || y >= input.size_[1])
    {
        sdata[tid] = 0;
        return;
    }

    const int Nw = weights.size_[0];

    const T k = ((vmax - vmin) / (Nw - 1));

    const T grad_out_pos = grad_output(x, y);
    const T inp_pos = input(x, y);
    const T idx = (inp_pos - vmin) / k;
    T grad_inp = 0;
    bool boundary = false;
    for (int i = 0; i < Nw; ++i)
    {
        // perform extrapolation
        if (i == 0 && idx <= i)
        {
            sdata[tid] = grad_out_pos;
            boundary = true;
        }
        else if (i == Nw -1 && idx >= i)
        {
            sdata[tid] = grad_out_pos;
            boundary = true;
        }
        else if (!boundary && idx >= i && idx < i + 1)
        {
            const T alpha = idx - i;
            sdata[tid] = (1 - alpha) * grad_out_pos;
            grad_inp += weights(i, y) * (-grad_out_pos/k);
        }
        else if (!boundary && idx >= i - 1 && idx < i)
        {
            const T alpha = idx - i + 1;
            sdata[tid] = alpha * grad_out_pos;
            grad_inp += weights(i, y) * (grad_out_pos/k);
        }
        else
            sdata[tid] = 0;

        // parallel reduction along outer dimensions
        __syncthreads();

        parallelReduce(sdata, tid, blockDim.x);

        if(tid == 0)
            atomicAdd(&(grad_weights(i, y)), sdata[tid]);
    }
    grad_input(x, y) = grad_inp;
}


template<typename T>
void optox::LinearActOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);

    auto output = this->template getOutput<T, 2>(0, outputs);

    this->checkSize(input->size(), weights->size());

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    act_linear_forward_kernel<T><<<block_count, dim_block, 0, this->stream_>>>(
        *output,
        *input, *weights,
        this->vmin_, this->vmax_);
    OPTOX_CUDA_CHECK;
}

template<typename T>
void optox::LinearActOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);
    auto grad_output = this->template getInput<T, 2>(2, inputs);

    auto grad_input = this->template getOutput<T, 2>(0, outputs);
    auto grad_weights = this->template getOutput<T, 2>(1, outputs);

    this->checkSize(input->size(), weights->size());

    // clear the weights gradient
    grad_weights->fill(0);

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    act_linear_backward_kernel<T><<<block_count, dim_block, thread_per_block * sizeof(T), this->stream_>>>(
        *grad_input, *grad_weights,
        *input, *weights, *grad_output,
        this->vmin_, this->vmax_);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP(T) \
    template class optox::LinearActOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
