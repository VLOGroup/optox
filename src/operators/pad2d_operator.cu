///@file pad2d_operator.cu
///@brief Operator that pads an image given with symmetric boundary conndition
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.202


#include "utils.h"
#include "tensor/d_tensor.h"
#include "pad2d_operator.h"

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
__global__ void pad2d(
    typename optox::DTensor<T, 3>::Ref out,
    const typename optox::DTensor<T, 3>::ConstRef in,
    int left, int top)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < out.size_[2] && y < out.size_[1] && z < out.size_[0])
    {
        // compute the corresponding index 
        const int x_in = getPixel<M>(x - left, in.size_[2]);
        const int y_in = getPixel<M>(y - top, in.size_[1]);
        out(z, y, x) = in(z, y_in, x_in);
    }
}


template<typename T>
void optox::Pad2dOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 3>(0, inputs);
    auto out = this->template getOutput<T, 3>(0, outputs);

    if (x->size()[0] != out->size()[0] || 
        x->size()[1]+this->top_+this->bottom_ != out->size()[1]||
        x->size()[2]+this->left_+this->right_ != out->size()[2])
        THROW_OPTOXEXCEPTION("Pad2dOperator: input and output size do not match!");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(out->size()[2], dim_block.x),
                         divUp(out->size()[1], dim_block.y),
                         divUp(out->size()[0], dim_block.z));

    switch (mode_)
    {
        case optox::PaddingMode::symmetric:
            pad2d<T, optox::PaddingMode::symmetric> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_);
            break;
        case optox::PaddingMode::reflect:
            pad2d<T, optox::PaddingMode::reflect> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_);
            break;
        case optox::PaddingMode::replicate:
            pad2d<T, optox::PaddingMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, 
                this->left_, this->top_);
            break;
    }
    OPTOX_CUDA_CHECK;
}


template <typename T, optox::PaddingMode M>
__global__ void pad2d_grad(
    typename optox::DTensor<T, 3>::Ref grad_in,
    const typename optox::DTensor<T, 3>::ConstRef grad_out,
    int left, int top)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < grad_out.size_[2] && y < grad_out.size_[1] && z < grad_out.size_[0])
    {
        // compute the corresponding index 
        const int x_in = getPixel<M>(x - left, grad_in.size_[2]);
        const int y_in = getPixel<M>(y - top, grad_in.size_[1]);
        atomicAdd(&grad_in(z, y_in, x_in), grad_out(z, y, x));
    }
}


template<typename T>
void optox::Pad2dOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_out = this->template getInput<T, 3>(0, inputs);

    auto grad_x = this->template getOutput<T, 3>(0, outputs);

    // clear the weights gradient
    grad_x->fill(0);

    if (grad_x->size()[0] != grad_out->size()[0] || 
        grad_x->size()[1]+this->top_+this->bottom_ != grad_out->size()[1]||
        grad_x->size()[2]+this->left_+this->right_ != grad_out->size()[2])
        THROW_OPTOXEXCEPTION("Pad2dOperator-adjoint: input and output size do not match!");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(grad_out->size()[2], dim_block.x),
                         divUp(grad_out->size()[1], dim_block.y),
                         divUp(grad_out->size()[0], dim_block.z));

    switch (mode_)            
    {
        case optox::PaddingMode::symmetric:
            pad2d_grad<T,optox::PaddingMode::symmetric> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_);
            break;
        case optox::PaddingMode::reflect:
            pad2d_grad<T,optox::PaddingMode::reflect> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_);
            break;
        case optox::PaddingMode::replicate:
            pad2d_grad<T,optox::PaddingMode::replicate> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *grad_out, 
                this->left_, this->top_);
            break;
    }
    OPTOX_CUDA_CHECK;
}

#define REGISTER_OP(T) \
    template class optox::Pad2dOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
