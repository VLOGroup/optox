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

    if (ix < x.size_[0] && iy < x.size_[1])
    {
        const int xp = ix + (ix < x.size_[0] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);

        y(ix, iy, 0) = x(xp, iy, 0) - x(ix, iy, 0);
        y(ix, iy, 1) = x(xp, iy, 1) - x(ix, iy, 1);
        y(ix, iy, 2) = x(ix, yp, 0) - x(ix, iy, 0);
        y(ix, iy, 3) = x(ix, yp, 1) - x(ix, iy, 1);
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

    if (ix < x.size_[0] && iy < x.size_[1] && iz < x.size_[2])
    {
        const int xp = ix + (ix < x.size_[0] - 1);
        const int yp = iy + (iy < x.size_[1] - 1);
        const int zp = iz + (iz < x.size_[2] - 1);

        y(ix, iy, iz, 0) = x(xp, iy, iz, 0) - x(ix, iy, iz, 0);
        y(ix, iy, iz, 1) = x(xp, iy, iz, 1) - x(ix, iy, iz, 1);
        y(ix, iy, iz, 2) = x(xp, iy, iz, 2) - x(ix, iy, iz, 2);
        y(ix, iy, iz, 3) = x(ix, yp, iz, 0) - x(ix, iy, iz, 0);
        y(ix, iy, iz, 4) = x(ix, yp, iz, 1) - x(ix, iy, iz, 1);
        y(ix, iy, iz, 5) = x(ix, yp, iz, 2) - x(ix, iy, iz, 2);
        y(ix, iy, iz, 6) = x(ix, iy, zp, 0) - x(ix, iy, iz, 0);
        y(ix, iy, iz, 7) = x(ix, iy, zp, 1) - x(ix, iy, iz, 1);
        y(ix, iy, iz, 8) = x(ix, iy, zp, 2) - x(ix, iy, iz, 2);
    }
}

template<typename T, unsigned int N>
void optox::Nabla2Operator<T, N>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, N+1>(0, inputs);
    auto y = this->template getOutput<T, N+1>(0, outputs);

    if (x->size()[N] != N || y->size()[N] != N*N)
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported size");

    dim3 dim_block;
    if (N == 2)
        dim_block = dim3(32, 32);
    else if (N == 3)
        dim_block = dim3(16, 16, 3);
    else
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported dimension");

    dim3 dim_grid(divUp(x->size()[0], dim_block.x),
                  divUp(x->size()[1], dim_block.y),
                  divUp(x->size()[2], dim_block.z));

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

    if (ix < x.size_[0] && iy < x.size_[1])
    {
        T div_xx_x = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, 0) + y(ix - 1, iy, 0)
                                            :
                                            y(ix - 1, iy, 0)
                        :
                        -y(ix, iy, 0);

        T div_xy_x = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, 1) + y(ix - 1, iy, 1)
                                            :
                                            y(ix - 1, iy, 1)
                        :
                        -y(ix, iy, 1);

        T div_yx_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, 2) + y(ix, iy - 1, 2)
                                            :
                                            y(ix, iy - 1, 2)
                        :
                        -y(ix, iy, 2);

        T div_yy_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, 3) + y(ix, iy - 1, 3)
                                            :
                                            y(ix, iy - 1, 3)
                        :
                        -y(ix, iy, 3);

        x(ix, iy, 0) = div_xx_x + div_yx_y;
        x(ix, iy, 1) = div_yy_y + div_xy_x;
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
  
    if (ix < x.size_[0] && iy < x.size_[1] && iz < x.size_[2])
    {
        T div_xx_x = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, iz, 0) + y(ix - 1, iy, iz, 0)
                                            :
                                            y(ix - 1, iy, iz, 0)
                        :
                        -y(ix, iy, iz, 0);
        T div_xy_x = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, iz, 1) + y(ix - 1, iy, iz, 1)
                                            :
                                            y(ix - 1, iy, iz, 1)
                        :
                        -y(ix, iy, iz, 1);
        T div_xz_x = (ix > 0) ? 
                        (ix < x.size_[0] - 1) ?
                                            -y(ix, iy, iz, 2) + y(ix - 1, iy, iz, 2)
                                            :
                                            y(ix - 1, iy, iz, 2)
                        :
                        -y(ix, iy, iz, 2);

        T div_yx_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, iz, 3) + y(ix, iy - 1, iz, 3)
                                            :
                                            y(ix, iy - 1, iz, 3)
                        :
                        -y(ix, iy, iz, 3);
        T div_yy_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, iz, 4) + y(ix, iy - 1, iz, 4)
                                            :
                                            y(ix, iy - 1, iz, 4)
                        :
                        -y(ix, iy, iz, 4);
        T div_yz_y = (iy > 0) ? 
                        (iy < x.size_[1] - 1) ?
                                            -y(ix, iy, iz, 5) + y(ix, iy - 1, iz, 5)
                                            :
                                            y(ix, iy - 1, iz, 5)
                        :
                        -y(ix, iy, iz, 5);
        
        T div_zx_z = (iz > 0) ? 
                        (iz < x.size_[2] - 1) ?
                                            -y(ix, iy, iz, 6) + y(ix, iy, iz - 1, 6)
                                            :
                                            y(ix, iy, iz - 1, 6)
                        :
                        -y(ix, iy, iz, 6);
        T div_zy_z = (iz > 0) ? 
                        (iz < x.size_[2] - 1) ?
                                            -y(ix, iy, iz, 7) + y(ix, iy, iz - 1, 7)
                                            :
                                            y(ix, iy, iz - 1, 7)
                        :
                        -y(ix, iy, iz, 7);
        T div_zz_z = (iz > 0) ? 
                        (iz < x.size_[2] - 1) ?
                                            -y(ix, iy, iz, 8) + y(ix, iy, iz - 1, 8)
                                            :
                                            y(ix, iy, iz - 1, 8)
                        :
                        -y(ix, iy, iz, 8);

        x(ix, iy, iz, 0) = div_xx_x + div_yx_y + div_zx_z;
        x(ix, iy, iz, 1) = div_xy_x + div_yy_y + div_zy_z;
        x(ix, iy, iz, 2) = div_xz_x + div_yz_y + div_zz_z;
    }
}

template<typename T, unsigned int N>
void optox::Nabla2Operator<T, N>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto y = this->template getInput<T, N+1>(0, inputs);
    auto x = this->template getOutput<T, N+1>(0, outputs);

    if (x->size()[N] != N || y->size()[N] != N*N)
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported size");

    dim3 dim_block;
    if (N == 2)
        dim_block = dim3(32, 32);
    else if (N == 3)
        dim_block = dim3(16, 16, 3);
    else
        THROW_OPTOXEXCEPTION("Nabla2Operator: unsupported dimension");

    dim3 dim_grid(divUp(x->size()[0], dim_block.x),
                  divUp(x->size()[1], dim_block.y),
                  divUp(x->size()[2], dim_block.z));

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
