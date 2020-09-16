///@file pad3d_operator.cu
///@brief Operator that pads an image given with symmetric boundary conndition
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 04.2020

#include "utils.h"
#include "tensor/d_tensor.h"
#include "pad3d_operator.h"

#include "reduce.cuh"


template <optox::PaddingMode M>
inline __device__ int getPixel(int x, int width)
{
    int x_ = x;
    switch (M)
    {
        case optox::PaddingMode::symmetric:
            if (x < 0)
                x_ = abs(x) - 1;
            else if (x >= width)
                x_ = 2 * width - x - 1;
            break;
        case optox::PaddingMode::reflect:
            if (x < 0)
                x_ = abs(x);
            else if (x >= width)
                x_ = 2 * width - x - 2;
            break;
        case optox::PaddingMode::replicate:
            if (x < 0)
                x_ = 0;
            else if (x >= width)
                x_ = width - 1;
            break;
    }
    return x_;
}


template <typename T, optox::PaddingMode M>
__global__ void pad3d(
    typename optox::DTensor<T, 4>::Ref out,
    const typename optox::DTensor<T, 4>::ConstRef in,
    int left, int top, int front)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int N = threadIdx.z + blockIdx.z * blockDim.z;
    int z = N / out.size_[0];
    int n = N % out.size_[0];

    if (x < out.size_[3] && y < out.size_[2] && z < out.size_[1] && n < out.size_[0])
    {
        // compute the corresponding index 
        const int x_in = getPixel<M>(x - left, in.size_[3]);
        const int y_in = getPixel<M>(y - top, in.size_[2]);
        const int z_in = getPixel<M>(z - front, in.size_[1]);
        out(n, z, y, x) = in(n, z_in, y_in, x_in);
    }
}


template<typename T>
void optox::Pad3dOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 4>(0, inputs);
    auto out = this->template getOutput<T, 4>(0, outputs);

    if (x->size()[0] != out->size()[0] || 
        x->size()[1]+this->paddingZ()  != out->size()[1] ||
        x->size()[2]+this->paddingY() != out->size()[2]||
        x->size()[3]+this->paddingX() != out->size()[3])
        THROW_OPTOXEXCEPTION("Pad3dOperator: input and output size do not match!");

    dim3 dim_block = dim3(16, 16, 4);
    dim3 dim_grid = dim3(divUp(out->size()[3], dim_block.x),
                         divUp(out->size()[2], dim_block.y),
                         divUp(out->size()[0]*out->size()[1], dim_block.z));

    switch (mode_)
    {
        case optox::PaddingMode::symmetric:
            pad3d<T, optox::PaddingMode::symmetric> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_, this->front_);
            break;
        case optox::PaddingMode::reflect:
            pad3d<T, optox::PaddingMode::reflect> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_, this->front_);
            break;
        case optox::PaddingMode::replicate:
            pad3d<T, optox::PaddingMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_, this->front_);
            break;
    }
    OPTOX_CUDA_CHECK;
}


template <typename T, optox::PaddingMode M>
__global__ void pad3d_grad(
    typename optox::DTensor<T, 4>::Ref grad_in,
    const typename optox::DTensor<T, 4>::ConstRef grad_out,
    int left, int top, int front)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int N = threadIdx.z + blockIdx.z * blockDim.z;
    int z = N / grad_out.size_[0];
    int n = N % grad_out.size_[0];

    if (x < grad_out.size_[3] && y < grad_out.size_[2] && z < grad_out.size_[1] && n < grad_out.size_[0])
    {
        // compute the corresponding index 
        const int x_in = getPixel<M>(x - left, grad_in.size_[3]);
        const int y_in = getPixel<M>(y - top, grad_in.size_[2]);
        const int z_in = getPixel<M>(z - front, grad_in.size_[1]);
        atomicAdd(&grad_in(n, z_in, y_in, x_in), grad_out(n, z, y, x));
    }
}


template<typename T>
void optox::Pad3dOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_out = this->template getInput<T, 4>(0, inputs);
    auto grad_x = this->template getOutput<T, 4>(0, outputs);

    // clear the weights gradient
    grad_x->fill(0);

    if (grad_x->size()[0] != grad_out->size()[0] || 
        grad_x->size()[1]+this->paddingZ() != grad_out->size()[1]||
        grad_x->size()[2]+this->paddingY() != grad_out->size()[2]||
        grad_x->size()[3]+this->paddingX() != grad_out->size()[3])
        THROW_OPTOXEXCEPTION("Pad3dOperator-adjoint: input and output size do not match!");

    dim3 dim_block = dim3(16, 16, 4);
    dim3 dim_grid = dim3(divUp(grad_out->size()[3], dim_block.x),
                         divUp(grad_out->size()[2], dim_block.y),
                         divUp(grad_out->size()[0]*grad_out->size()[1], dim_block.z));

    switch (mode_)            
    {
        case optox::PaddingMode::symmetric:
            pad3d_grad<T,optox::PaddingMode::symmetric> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_, this->front_);
            break;
        case optox::PaddingMode::reflect:
            pad3d_grad<T,optox::PaddingMode::reflect> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_, this->front_);
            break;
        case optox::PaddingMode::replicate:
            pad3d_grad<T,optox::PaddingMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_, this->front_);
            break;
    }
    OPTOX_CUDA_CHECK;
}

#define REGISTER_OP(T) \
    template class optox::Pad3dOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
