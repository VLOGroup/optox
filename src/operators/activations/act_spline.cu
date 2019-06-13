///@file act_spline.cu
///@brief Spline basis function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "act_spline.h"
#include "reduce.cuh"


// b-spline quadratic Activation Functor
inline __device__ float b_spline_quad(float x)
{
    x = fabs(x);

    if (x <= 0.5f) 
        return 0.75f - x*x;
    else if (x <= 1.5f) 
        return (1.5f - x)*(1.5f - x)*0.5f;
    else 
        return 0.f;
}
inline __device__ double b_spline_quad(double x)
{
    x = abs(x);

    if (x <= 0.5) 
        return 0.75 - x*x;
    else if (x <= 1.5) 
        return (1.5 - x)*(1.5 - x)*0.5;
    else 
        return 0.;
}

// first derivative of quadratic spline base function
inline __device__ float b_spline_quad_prime(float x)
{
  if (-1.5f <= x && x < -0.5f) return x + 1.5f;
  else if (-0.5f <= x && x <= 0.5f) return -2*x;
  else if (0.5f <= x && x <= 1.5f) return x - 1.5f;
  else return 0.f;
}

inline __device__ double b_spline_quad_prime(double x)
{
  if (-1.5 <= x && x < -0.5) return x + 1.5;
  else if (-0.5 <= x && x <= 0.5) return -2*x;
  else if (0.5 <= x && x <= 1.5) return x - 1.5;
  else return 0.;
}

template<typename T>
__global__ void act_spline_forward_kernel(
    typename optox::DTensor<T, 2>::Ref output,
    const typename optox::DTensor<T, 2>::ConstRef input,
    const typename optox::DTensor<T, 2>::ConstRef weights,
    T vmin, T vmax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= input.size_[1] || y >= input.size_[0])
        return;

    const int Nw = weights.size_[1];
    const T sigma = (vmax - vmin) / (Nw - 1);

    const T inp_pos = input(y, x);
    T val = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function
        const T mu = sigma * i + vmin;
        const T diff = (inp_pos - mu) / sigma;
        const T base_function = b_spline_quad(diff);
        val += weights(y, i) * base_function;
    }

    output(y, x) = val;
}


template<typename T>
__global__ void act_spline_backward_kernel(
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

    if (x >= input.size_[1] || y >= input.size_[0])
    {
        sdata[tid] = 0;
        return;
    }

    const int Nw = weights.size_[1];
    const T sigma = (vmax - vmin) / (Nw - 1);

    const T inp_pos = input(y, x);
    const T grad_out_pos = grad_output(y, x);
    T grad_inp = 0;
    for (int i = 0; i < Nw; ++i)
    {
        // compute the base function
        const T mu = sigma * i + vmin;
        const T diff = (inp_pos - mu) / sigma;
        const T base_function = b_spline_quad(diff);
        const T base_function_prime = b_spline_quad_prime(diff) / sigma;
        // backpropagate the gradient to the input
        grad_inp += weights(y, i) * base_function_prime * grad_out_pos;

        // backpropagate the gradient to a single weight
        sdata[tid] = base_function * grad_out_pos;

        // parallel reduction along outer dimensions
        __syncthreads();

        parallelReduce(sdata, tid, blockDim.x);

        if(tid == 0)
            atomicAdd(&(grad_weights(y, i)), sdata[tid]);
    }
    grad_input(y, x) = grad_inp;
}


template<typename T>
void optox::SplineActOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 2>(0, inputs);
    auto weights = this->template getInput<T, 2>(1, inputs);

    auto output = this->template getOutput<T, 2>(0, outputs);

    this->checkSize(input->size(), weights->size());

    int thread_per_block = 256;
    dim3 dim_block = dim3(thread_per_block, 1);
    dim3 block_count = dim3(divUp(input->size()[1], dim_block.x),
                            input->size()[0]);

    act_spline_forward_kernel<T><<<block_count, dim_block, 0, this->stream_>>>(
        *output,
        *input, *weights,
        this->vmin_, this->vmax_);
    OPTOX_CUDA_CHECK;
}

template<typename T>
void optox::SplineActOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
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
    dim3 block_count = dim3(divUp(input->size()[1], dim_block.x),
                            input->size()[0]);

    act_spline_backward_kernel<T><<<block_count, dim_block, thread_per_block * sizeof(T), this->stream_>>>(
        *grad_input, *grad_weights,
        *input, *weights, *grad_output,
        this->vmin_, this->vmax_);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP(T) \
    template class optox::SplineActOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
