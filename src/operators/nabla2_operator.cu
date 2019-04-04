///@file nabla2_operator.cu
///@brief Operator that computes the second order forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "nabla2_operator.h"

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 3>::Ref y,
    const typename optox::DTensor<T, 3>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[2] && iy < x.size_[1])
    {
        const int xp = ix + (ix < x.size_[2] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);

        y(0, iy, ix) = x(0, iy, xp) - x(0, iy, ix);
        y(1, iy, ix) = x(1, iy, xp) - x(1, iy, ix);
        y(2, iy, ix) = x(0, yp, ix) - x(0, iy, ix);
        y(3, iy, ix) = x(1, yp, ix) - x(1, iy, ix);
    }
}

template<typename T>
__global__ void forward_differences(
    typename optox::DTensor<T, 4>::Ref y,
    const typename optox::DTensor<T, 4>::ConstRef x)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    if (ix < x.size_[3] && iy < x.size_[2] && iz < x.size_[1])
    {
        const int xp = ix + (ix < x.size_[3] - 1);
        const int yp = iy + (iy < x.size_[2] - 1);
        const int zp = iz + (iz < x.size_[1] - 1);

        y(0, iz, iy, ix) = x(0, iz, iy, xp) - x(0, iz, iy, ix);
        y(1, iz, iy, ix) = x(1, iz, iy, xp) - x(1, iz, iy, ix);
        y(2, iz, iy, ix) = x(2, iz, iy, xp) - x(2, iz, iy, ix);
        y(3, iz, iy, ix) = x(0, iz, yp, ix) - x(0, iz, iy, ix);
        y(4, iz, iy, ix) = x(1, iz, yp, ix) - x(1, iz, iy, ix);
        y(5, iz, iy, ix) = x(2, iz, yp, ix) - x(2, iz, iy, ix);
        y(6, iz, iy, ix) = x(0, zp, iy, ix) - x(0, iz, iy, ix);
        y(7, iz, iy, ix) = x(1, zp, iy, ix) - x(1, iz, iy, ix);
        y(8, iz, iy, ix) = x(2, zp, iy, ix) - x(2, iz, iy, ix);
    }
}

template<typename T, unsigned int N>
void optox::Nabla2Operator<T, N>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, N+1>(0, inputs);
    auto y = this->template getOutput<T, N+1>(0, outputs);

    if (x->size()[0] != N || y->size()[0] != N*N)
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported size");

    dim3 dim_block;
    dim3 dim_grid;
    if (N == 2)
    {
        dim_block = dim3(32, 32);
        dim_grid = dim3(divUp(x->size()[2], dim_block.x),
                        divUp(x->size()[1], dim_block.y));
    }
    else if (N == 3)
    {
        dim_block = dim3(16, 16, 3);
        dim_grid = dim3(divUp(x->size()[3], dim_block.x),
                        divUp(x->size()[2], dim_block.y),
                        divUp(x->size()[1], dim_block.z));
    }
    else
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported dimension");

    forward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*y, *x);
    OPTOX_CUDA_CHECK;
}


template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 3>::Ref x,
    const typename optox::DTensor<T, 3>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < x.size_[2] && iy < x.size_[1])
    {
        T div_xx_x = (ix > 0) ? 
                        (ix < x.size_[2] - 1) ?
                                            -y(0, iy, ix) + y(0, iy, ix - 1)
                                            :
                                            y(0, iy, ix - 1)
                        :
                        -y(0, iy, ix);

        T div_xy_x = (ix > 0) ? 
                        (ix < x.size_[2] - 1) ?
                                            -y(1, iy, ix) + y(1, iy, ix - 1)
                                            :
                                            y(1, iy, ix - 1)
                        :
                        -y(1, iy, ix);

        T div_yx_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(2, iy, ix) + y(2, iy - 1, ix)
                                            :
                                            y(2, iy - 1, ix)
                        :
                        -y(2, iy, ix);

        T div_yy_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(3, iy, ix) + y(3, iy - 1, ix)
                                            :
                                            y(3, iy - 1, ix)
                        :
                        -y(3, iy, ix);

        x(0, iy, ix) = div_xx_x + div_yx_y;
        x(1, iy, ix) = div_yy_y + div_xy_x;
    }
}

template<typename T>
__global__ void backward_differences(
    typename optox::DTensor<T, 4>::Ref x,
    const typename optox::DTensor<T, 4>::ConstRef y)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;
  
    if (ix < x.size_[3] && iy < x.size_[2] && iz < x.size_[1])
    {
        T div_xx_x = (ix > 0) ? 
                        (ix < x.size_[3] - 1) ?
                                            -y(0, iz, iy, ix) + y(0, iz, iy, ix - 1)
                                            :
                                            y(0, iz, iy, ix - 1)
                        :
                        -y(0, iz, iy, ix);
        T div_xy_x = (ix > 0) ? 
                        (ix < x.size_[3] - 1) ?
                                            -y(1, iz, iy, ix) + y(1, iz, iy, ix - 1)
                                            :
                                            y(1, iz, iy, ix - 1)
                        :
                        -y(1, iz, iy, ix);
        T div_xz_x = (ix > 0) ? 
                        (ix < x.size_[3] - 1) ?
                                            -y(2, iz, iy, ix) + y(2, iz, iy, ix - 1)
                                            :
                                            y(2, iz, iy, ix - 1)
                        :
                        -y(2, iz, iy, ix);

        T div_yx_y = (iy > 0) ? 
                        (iy < x.size_[2] - 1) ?
                                            -y(3, iz, iy, ix) + y(3, iz, iy - 1, ix)
                                            :
                                            y(3, iz, iy - 1, ix)
                        :
                        -y(3, iz, iy, ix);
        T div_yy_y = (iy > 0) ? 
                        (iy < x.size_[2] - 1) ?
                                            -y(4, iz, iy, ix) + y(4, iz, iy - 1, ix)
                                            :
                                            y(4, iz, iy - 1, ix)
                        :
                        -y(4, iz, iy, ix);
        T div_yz_y = (iy > 0) ? 
                        (iy < x.size_[2] - 1) ?
                                            -y(5, iz, iy, ix) + y(5, iz, iy - 1, ix)
                                            :
                                            y(5, iz, iy - 1, ix)
                        :
                        -y(5, iz, iy, ix);
        
        T div_zx_z = (iz > 0) ? 
                        (iz < x.size_[1] - 1) ?
                                            -y(6, iz, iy, ix) + y(6, iz - 1, iy, ix)
                                            :
                                            y(6, iz - 1, iy, ix)
                        :
                        -y(6, iz, iy, ix);
        T div_zy_z = (iz > 0) ? 
                        (iz < x.size_[1] - 1) ?
                                            -y(7, iz, iy, ix) + y(7, iz - 1, iy, ix)
                                            :
                                            y(7, iz - 1, iy, ix)
                        :
                        -y(7, iz, iy, ix);
        T div_zz_z = (iz > 0) ? 
                        (iz < x.size_[1] - 1) ?
                                            -y(8, iz, iy, ix) + y(8, iz - 1, iy, ix)
                                            :
                                            y(8, iz - 1, iy, ix)
                        :
                        -y(8, iz, iy, ix);

        x(0, iz, iy, ix) = div_xx_x + div_yx_y + div_zx_z;
        x(1, iz, iy, ix) = div_xy_x + div_yy_y + div_zy_z;
        x(2, iz, iy, ix) = div_xz_x + div_yz_y + div_zz_z;
    }
}

template<typename T, unsigned int N>
void optox::Nabla2Operator<T, N>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto y = this->template getInput<T, N+1>(0, inputs);
    auto x = this->template getOutput<T, N+1>(0, outputs);

    if (x->size()[0] != N || y->size()[0] != N*N)
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported size");

    dim3 dim_block;
    dim3 dim_grid;
    if (N == 2)
    {
        dim_block = dim3(32, 32);
        dim_grid = dim3(divUp(x->size()[2], dim_block.x),
                        divUp(x->size()[1], dim_block.y));
    }
    else if (N == 3)
    {
        dim_block = dim3(16, 16, 3);
        dim_grid = dim3(divUp(x->size()[3], dim_block.x),
                        divUp(x->size()[2], dim_block.y),
                        divUp(x->size()[1], dim_block.z));
    }
    else
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported dimension");

    backward_differences<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*x, *y);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP_T(T, N) \
    template class optox::Nabla2Operator<T, N>;

#define REGISTER_OP(T) \
    REGISTER_OP_T(T, 2) \
    REGISTER_OP_T(T, 3)

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
#undef REGISTER_OP_T
