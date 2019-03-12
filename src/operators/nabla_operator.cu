///@file nabla_operator.cu
///@brief Operator that computes the forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018


#include "utils.h"
#include "tensor/d_tensor.h"
#include "nabla_operator.h"

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 3>::Ref y,
    const typename optox::DTensor<T, 2>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[0] && iy < x.size_[1])
    {

        const int xp = ix + (ix < x.size_[0] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);

        y(ix, iy, 0) = x(xp, iy) - x(ix, iy);
        y(ix, iy, 1) = x(ix, yp) - x(ix, iy);
    }
}

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 4>::Ref y,
    const typename optox::DTensor<T, 3>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
  
    if (ix < x.size_[0] && iy < x.size_[1] && iz < x.size_[2])
    {
        const int xp = ix + (ix < x.size_[0] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);
        const int zp = iz + (iz < x.size_[2] - 1);

        y(ix, iy, iz, 0) = x(xp, iy, iz) - x(ix, iy, iz);
        y(ix, iy, iz, 1) = x(ix, yp, iz) - x(ix, iy, iz);
        y(ix, iy, iz, 2) = x(ix, iy, zp) - x(ix, iy, iz);
    }
}

template<typename T, unsigned int N>
void optox::NablaOperator<T, N>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, N>(0, inputs);
    auto y = this->template getOutput<T, N+1>(0, outputs);

    if (y->size()[N] != N)
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported size");

    dim3 dim_block;
    if (N == 2)
        dim_block = dim3(32, 32);
    else if (N == 3)
        dim_block = dim3(16, 16, 3);
    else
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported dimension");

    dim3 dim_grid(divUp(x->size()[0], dim_block.x),
                  divUp(x->size()[1], dim_block.y),
                  divUp(x->size()[2], dim_block.z));

    forward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*y, *x);
    OPTOX_CUDA_CHECK;
}


template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 2>::Ref x,
    const typename optox::DTensor<T, 3>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[0] && iy < x.size_[1])
    {
        T div = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, 0) + y(ix - 1, iy, 0)
                                            :
                                            y(ix - 1, iy, 0)
                        :
                        -y(ix, iy, 0);

        div += (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, 1) + y(ix, iy - 1, 1)
                                            :
                                            y(ix, iy - 1, 1)
                        :
                        -y(ix, iy, 1);

        x(ix, iy) = div;
    }
}

template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 3>::Ref x,
    const typename optox::DTensor<T, 4>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
  
    if (ix < x.size_[0] && iy < x.size_[1] && iz < x.size_[2])
    {
        T div = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, iz, 0) + y(ix - 1, iy, iz, 0)
                                            :
                                            y(ix - 1, iy, iz, 0)
                        :
                        -y(ix, iy, iz, 0);

        div += (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, iz, 1) + y(ix, iy - 1, iz, 1)
                                            :
                                            y(ix, iy - 1, iz, 1)
                        :
                        -y(ix, iy, iz, 1);

        div += (iz > 0) ? 
                        (iz < x.size_[2] - 1) ?
                                            -y(ix, iy, iz, 2) + y(ix, iy, iz - 1, 2)
                                            :
                                            y(ix, iy, iz - 1, 2)
                        :
                        -y(ix, iy, iz, 2);

        x(ix, iy, iz) = div;
    }
}

template<typename T, unsigned int N>
void optox::NablaOperator<T, N>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto y = this->template getInput<T, N+1>(0, inputs);
    auto x = this->template getOutput<T, N>(0, outputs);

    if (y->size()[N] != N)
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported size");

    dim3 dim_block;
    if (N == 2)
        dim_block = dim3(32, 32);
    else if (N == 3)
        dim_block = dim3(16, 16, 3);
    else
        THROW_OPTOXEXCEPTION("NablaOperator: unsupported dimension");

    dim3 dim_grid(divUp(x->size()[0], dim_block.x),
                  divUp(x->size()[1], dim_block.y),
                  divUp(x->size()[2], dim_block.z));

    backward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*x, *y);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP_T(T, N) \
    template class optox::NablaOperator<T, N>;;

#define REGISTER_OP(T) \
    REGISTER_OP_T(T, 2) \
    REGISTER_OP_T(T, 3)

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
#undef REGISTER_OP_T
