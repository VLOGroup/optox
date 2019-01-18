///@file act_rbf.cu
///@brief Gaussian radial basis function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019


#include <iu/iucore.h>
#include <iu/iumath.h>
#include <type_traits>

#include "act_rbf.h"
#include "utils.cuh"


// forward Gaussian rbf
template<typename T>
__global__ void act_rbf_forward_kernel(
    typename iu::LinearDeviceMemory<T, 2>::KernelData output,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData input,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData weights,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.size_[0] || y >= input.size_[1])
        return;

    const int Nw = weights.size_[0];

    const T sigma = (vmax - vmin) / (Nw - 1);
    const T sigma2 = sigma * sigma;
    const T k = ((vmax - vmin) / (Nw - 1));

    T inp_pos = input(x, y);
    T val = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function
        const T mu = k * i + vmin;
        T base_function = 0;
        const T diff = inp_pos - mu;
        if (std::is_same<T, float>::value)
            base_function = expf( -(diff*diff) / (sigma2 * 2)) * 0.4;
        else
            base_function = exp( -(diff*diff) / (sigma2 * 2)) * 0.4;
        val += weights(i, y) * base_function;
    }

    output(x, y) = val;
}


// backward Gaussian rbf
template<typename T>
__global__ void act_rbf_backward_kernel(
    typename iu::LinearDeviceMemory<T, 2>::KernelData grad_input,
    typename iu::LinearDeviceMemory<T, 2>::KernelData grad_weights,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData input,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData weights,
    const typename iu::LinearDeviceMemory<T, 2>::KernelData grad_output,
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

    const T sigma = (vmax - vmin) / (Nw - 1);
    const T sigma2 = sigma * sigma;
    const T k = ((vmax - vmin) / (Nw - 1));

    T inp_pos = input(x, y);
    T grad_out_pos = grad_output(x, y);
    T grad_inp = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function and its derivative
        const T mu = k * i + vmin;
        T base_function = 0;
        T base_function_prime = 0;
        const T diff = inp_pos - mu;
        if (std::is_same<T, float>::value)
            base_function = expf( -(diff*diff) / (sigma2 * 2)) * 0.4;
        else
            base_function = exp( -(diff*diff) / (sigma2 * 2)) * 0.4;
        base_function_prime = base_function * (-diff)/sigma2;
        // backpropagate the gradient to the input
        grad_inp += weights(i, y) * base_function_prime;

        // backpropagate the gradient to a single weight
        sdata[tid] = base_function * grad_out_pos;

        // parallel reduction along outer dimensions
        parallelReduce(sdata, tid, blockDim.x);

        if(tid == 0)
            atomicAdd(&(grad_weights(i, y)), sdata[tid]);
    }
    grad_input(x, y) = grad_inp;
}


template<typename T>
void optox::RBFActOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);

    auto output = this->template getOutput<T, 2>(0, outputs);

    this->checkSize(input->size(), weights->size());

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(iu::divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    std::cout << "input " << *input << std::endl;
    std::cout << "weights " << *weights << std::endl;
    act_rbf_forward_kernel<T><<<dim_block, block_count, 0, this->stream_>>>(
        *output,
        *input, *weights,
        this->vmin_, this->vmax_);
}

template<typename T>
void optox::RBFActOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);
    auto grad_output = this->template getInput<T, 2>(2, inputs);

    auto grad_input = this->template getOutput<T, 2>(0, outputs);
    auto grad_weights = this->template getOutput<T, 2>(1, outputs);

    this->checkSize(input->size(), weights->size());

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(iu::divUp(input->size()[0], dim_block.x),
                            input->size()[1]);

    act_rbf_backward_kernel<T><<<dim_block, block_count, thread_per_block * sizeof(T), this->stream_>>>(
        *grad_input, *grad_weights,
        *input, *weights, *grad_output,
        this->vmin_, this->vmax_);
}


#define REGISTER_OP(T) \
    template class optox::RBFActOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
