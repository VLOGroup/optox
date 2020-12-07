///@file warp_operator.cu
///@brief Operator that warps an image given a flow field
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "warp_operator.h"
#include "reduce.cuh"


template <typename T>
__global__ void warp(
    typename optox::DTensor<T, 4>::Ref out,
    const typename optox::DTensor<T, 4>::ConstRef x,
    const typename optox::DTensor<T, 4>::ConstRef u,
    int is)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < x.size_[3] && iy < x.size_[2] && iz < x.size_[1])
    {
        // first get the flow
        const T dx = u(is, iy, ix, 0);
        const T dy = u(is, iy, ix, 1);

        // compute the interpolation coefficients
        const int ix_f = floorf(ix + dx);
        const int iy_f = floorf(iy + dy);

        const int ix_c = ix_f + 1;
        const int iy_c = iy_f + 1;

        const T w = ix - ix_f;
        const T h = iy - iy_f;

        T i_ff = 0, i_fc = 0;
        if (ix_f >= 0 && ix_f < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
                i_ff = x(is, iz, iy_f, ix_f);

            if (iy_c >= 0 && iy_c < x.size_[2])
                i_fc = x(is, iz, iy_c, ix_f);
        }

        T i_cf = 0, i_cc = 0;
        if (ix_c >= 0 && ix_c < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
                i_cf = x(is, iz, iy_f, ix_c);

            if (iy_c >= 0 && iy_c < x.size_[2])
                i_cc = x(is, iz, iy_c, ix_c);
        }

        // compute the interpolated output
        out(is, iz, iy, ix) = (1 - h) * (1 - w) * i_ff +
                              (1 - h) * w * i_cf +
                              h * (1 - w) * i_fc + 
                              h * w * i_cc;
    }
}


template<typename T>
void optox::WarpOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 4>(0, inputs);
    auto u = this->template getInput<T, 4>(1, inputs);
    auto out = this->template getOutput<T, 4>(0, outputs);

    if (x->size() != out->size() || 
        x->size()[0] != u->size()[0] || x->size()[2] != u->size()[1] || 
        x->size()[3] != u->size()[2] || u->size()[3] != 2)
        THROW_OPTOXEXCEPTION("WarpOperator: unsupported size");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(x->size()[3], dim_block.x),
                         divUp(x->size()[2], dim_block.y),
                         divUp(x->size()[1], dim_block.z));

    for (unsigned int s = 0; s < x->size()[0]; ++s)
        warp<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*out, *x, *u, s);
    OPTOX_CUDA_CHECK;
}


template <typename T>
__global__ void warp_grad(
    typename optox::DTensor<T, 4>::Ref grad_x,
    const typename optox::DTensor<T, 4>::ConstRef x,
    const typename optox::DTensor<T, 4>::ConstRef u,
    const typename optox::DTensor<T, 4>::ConstRef grad_out,
    int is)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    if (ix < x.size_[3] && iy < x.size_[2] && iz < x.size_[1])
    {
        // first get the flow
        const T dx = u(is, iy, ix, 0);
        const T dy = u(is, iy, ix, 1);

        // get the output gradient
        const T grad_val = grad_out(is, iz, iy, ix);

        // compute the interpolation coefficients
        const int ix_f = floorf(ix + dx);
        const int iy_f = floorf(iy + dy);

        const int ix_c = ix_f + 1;
        const int iy_c = iy_f + 1;

        const T w = ix - ix_f;
        const T h = iy - iy_f;

        if (ix_f >= 0 && ix_f < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
                atomicAdd(&grad_x(is, iz, iy_f, ix_f), (1 - h) * (1 - w) * grad_val);

            if (iy_c >= 0 && iy_c < x.size_[2])
                atomicAdd(&grad_x(is, iz, iy_c, ix_f), h * (1 - w) * grad_val);
        }

        if (ix_c >= 0 && ix_c < x.size_[3])
        {
            if (iy_f >= 0 && iy_f < x.size_[2])
                atomicAdd(&grad_x(is, iz, iy_f, ix_c), (1 - h) * w * grad_val);

            if (iy_c >= 0 && iy_c < x.size_[2])
                atomicAdd(&grad_x(is, iz, iy_c, ix_c), h * w * grad_val);
        }
    }
}


template<typename T>
void optox::WarpOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 4>(0, inputs);
    auto u = this->template getInput<T, 4>(1, inputs);
    auto grad_out = this->template getInput<T, 4>(2, inputs);

    auto grad_x = this->template getOutput<T, 4>(0, outputs);

    // clear the weights gradient
    grad_x->fill(0);

    if (x->size() != grad_x->size() || x->size() != grad_out->size() || 
        x->size()[0] != u->size()[0] || x->size()[2] != u->size()[1] || 
        x->size()[3] != u->size()[2] || u->size()[3] != 2)
        THROW_OPTOXEXCEPTION("WarpOperator: unsupported size");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid = dim3(divUp(x->size()[3], dim_block.x),
                         divUp(x->size()[2], dim_block.y),
                         divUp(x->size()[1], dim_block.z));

    for (unsigned int s = 0; s < x->size()[0]; ++s)
        warp_grad<T> <<<dim_grid, dim_block, 0, this->stream_>>>(*grad_x, *x, *u, *grad_out, s);
    OPTOX_CUDA_CHECK;
}

#define REGISTER_OP(T) \
    template class optox::WarpOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
