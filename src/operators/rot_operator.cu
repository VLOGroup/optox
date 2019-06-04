///@file rot_operator.cu
///@brief Operator rotating kernel stack
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019


#include "utils.h"
#include "tensor/d_tensor.h"
#include "rot_operator.h"

// template<typename T>
//  __device__ T interpolate_bilinear(T *in, T idx, T idy, int kernel_width, int kernel_height)
//  {
//     const int idx_f = floorf(idx);
//     const int idy_f = floorf(idy);

//     const int idx_c = idx_f + 1;
//     const int idy_c = idy_f + 1;

//     const T w = idx - idx_f;
//     const T h = idy - idy_f;

//     T i_ff = 0, i_fc = 0;
//     if (idx_f >= 0 && idx_f < kernel_width)
//     {
//         if (idy_f >= 0 && idy_f < kernel_height)
//             i_ff = in[idx_f+kernel_width*idy_f];

//         if (idy_c >= 0 && idy_c < kernel_height)
//             i_fc = in[idx_f+kernel_width*idy_c];
//     }

//     T i_cf = 0, i_cc = 0;
//     if (idx_c >= 0 && idx_c < kernel_width)
//     {
//         if (idy_f >= 0 && idy_f < kernel_height)
//             i_cf = in[idx_c+kernel_width*idy_f];

//         if (idy_c >= 0 && idy_c < kernel_height)
//             i_cc = in[idx_c+kernel_width*idy_c];
//     }

//     T out = (1 - h) * (1 - w) * i_ff;
//     out += (1 - h) * w * i_cf;
//     out += h * (1 - w) * i_fc;
//     out += h * w * i_cc;

//     return out;
//  }


template<typename T>
inline __device__ T interpolate_cubic(volatile T *in, T idx, int kernel_size)
{
    const int idx_f = floorf(idx);
    const int idx_f_1 = idx_f - 1;
    const int idx_c = idx_f+1;
    const int idx_c_1 = idx_c+1;

    // get the input values
    T i_f = 0;
    if (idx_f >= 0 && idx_f < kernel_size)
        i_f = in[idx_f];
    T i_f_1 = 0;
    if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
        i_f_1 = in[idx_f_1];
    T i_c = 0;
    if (idx_c >= 0 && idx_c < kernel_size)
        i_c = in[idx_c];
    T i_c_1 = 0;
    if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
        i_c_1 = in[idx_c_1];

    // determine the coefficients
    const T p_f = i_f;
    const T p_prime_f = (i_c - i_f_1) / 2;
    const T p_c = i_c;
    const T p_prime_c = (i_c_1 - i_f) / 2;

    const T a = 2*p_f - 2*p_c + p_prime_f + p_prime_c;
    const T b = -3*p_f + 3*p_c - 2*p_prime_f - p_prime_c;
    const T c = p_prime_f;
    const T d = p_f;

    const T u = idx - idx_f;

    T out = u*(u*(u*a+b)+c) + d;

    return out;
}

template<typename T>
 __device__ T interpolate_bicubic(volatile T *in, T idx, T idy, int width, int height)
{
    const int idy_f = floorf(idy);

    T buff_y[4];

    for (int dy = -1; dy < 3; ++dy)
    {
        const int c_idx_y = idy_f + dy;

        if (c_idx_y >= 0 && c_idx_y < height)
        buff_y[dy+1] = interpolate_cubic<T>(&(in[width*c_idx_y]), idx, width);
        else
        buff_y[dy+1] = 0;
    }

    T out = interpolate_cubic<T>(buff_y, idy - idy_f + 1, 4);

    return out;
}


template<typename T>
__global__ void rotate_kernel(
    const typename optox::DTensor<T, 4>::ConstRef x,
    const typename optox::DTensor<T, 1>::ConstRef angles,
    typename optox::DTensor<T, 5>::Ref out)
{
    const int idx_out = threadIdx.x;
    const int idy_out = threadIdx.y;
    const int idz_in = threadIdx.z + blockIdx.z * blockDim.z;

    const int ids = idz_in % x.size_[0];
    const int idf = idz_in / x.size_[0];

    const int kernel_height = x.size_[2];
    const int kernel_width = x.size_[3];

    extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
    T *x_shared = reinterpret_cast<T*>(s_buffer);

    // load data into shared memory (used to interpolate)
    x_shared[idx_out+kernel_width*idy_out] = x(ids, idf, idy_out, idx_out);

    __syncthreads();

    const int center_x = kernel_width / 2;
    const int center_y = kernel_height / 2;
    // compute the centralized coordinates
    const int idx_out_c = idx_out - center_x;
    const int idy_out_c = -idy_out + center_y;

    for (int angle_idx = 0; angle_idx < angles.size_[0]; ++angle_idx)
    {
        const T theta = angles(angle_idx);
        // rotate the coordinate into the source image (-theta)
        const T cos_theta = cos(theta);
        const T sin_theta = sin(theta);
        const T idx_in_c = cos_theta * idx_out_c + sin_theta * idy_out_c;
        const T idy_in_c = -sin_theta * idx_out_c + cos_theta * idy_out_c;

        // transfer them back into the input image (in and out have same size)
        const T idx_in = idx_in_c + center_x;
        const T idy_in = -idy_in_c + center_y;

        out(ids, angle_idx, idf, idy_out, idx_out) = interpolate_bicubic<T>(x_shared, idx_in,
            idy_in, kernel_width, kernel_height);
    }
}


template<typename T>
void optox::RotOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 4>(0, inputs);
    auto angles = this->template getInput<T, 1>(1, inputs);
    auto y = this->template getOutput<T, 5>(0, outputs);

    if (angles->size()[0] != y->size()[1])
        THROW_OPTOXEXCEPTION("RotOperator: angles and output require same size");

    dim3 block_size(x->size()[3],
                    x->size()[2]);
    dim3 grid_size(1, 1, x->size()[0]*x->size()[1]);

    unsigned int smem_size = x->size()[2] * x->size()[3] * sizeof(T);
    rotate_kernel<T><<<grid_size, block_size, smem_size, this->stream_>>>(*x, *angles, *y);
    OPTOX_CUDA_CHECK;
}


// template<typename T>
// inline __device__ void backpolate_bilinear(T *out, T val, T idx, T idy, int kernel_width, int kernel_height)
// {
//     const int idx_f = floor(idx);
//     const int idy_f = floor(idy);

//     const int idx_c = idx_f + 1;
//     const int idy_c = idy_f + 1;

//     const T w = idx - idx_f;
//     const T h = idy - idy_f;

//     if (idx_f >= 0 && idx_f < kernel_width)
//     {
//         if (idy_f >= 0 && idy_f < kernel_height)
//             atomicAdd(out + idx_f+kernel_width*idy_f, (1 - h) * (1 - w) * val);

//         if (idy_c >= 0 && idy_c < kernel_height)
//             atomicAdd(out + idx_f+kernel_width*idy_c, h * (1 - w) * val);
//     }

//     if (idx_c >= 0 && idx_c < kernel_width)
//     {
//         if (idy_f >= 0 && idy_f < kernel_height)
//             atomicAdd(out + idx_c+kernel_width*idy_f, (1 - h) * w * val);

//         if (idy_c >= 0 && idy_c < kernel_height)
//             atomicAdd(out + idx_c+kernel_width*idy_c, h * w * val);
//     }
// }


template<typename T>
inline __device__ void backpolate_cubic(T *out, T error, T idx, int kernel_size, bool direct_buf = false)
{
    const int idx_f = floor(idx);
    const int idx_f_1 = idx_f - 1;
    const int idx_c = idx_f+1;
    const int idx_c_1 = idx_c+1;

    const T u = idx - idx_f;
    const T uu = u*u;
    const T uuu = uu*u;

    // determine the coefficients
    T d_out_d_p_f_1 = -uuu/2 + uu - u/2;
    T d_out_d_p_f = (3*uuu)/2 - (5*uu)/2 + 1;
    T d_out_d_p_c = -(3*uuu)/2 + 2*uu + u/2;
    T d_out_d_p_c_1 = uuu/2 - uu/2;

    if (not direct_buf)
    {
        if (idx_f >= 0 && idx_f < kernel_size)
            atomicAdd(out + idx_f,   d_out_d_p_f   * error);
        if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
            atomicAdd(out + idx_f_1, d_out_d_p_f_1 * error);
        if (idx_c >= 0 && idx_c < kernel_size)
            atomicAdd(out + idx_c,   d_out_d_p_c   * error);
        if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
            atomicAdd(out + idx_c_1, d_out_d_p_c_1 * error);
    }
    else
    {
        if (idx_f >= 0 && idx_f < kernel_size)
            out[idx_f]   = d_out_d_p_f   * error;
        if (idx_f_1 >= 0 && idx_f_1 < kernel_size)
            out[idx_f_1] = d_out_d_p_f_1 * error;
        if (idx_c >= 0 && idx_c < kernel_size)
            out[idx_c]   = d_out_d_p_c   * error;
        if (idx_c_1 >= 0 && idx_c_1 < kernel_size)
            out[idx_c_1] = d_out_d_p_c_1 * error;
    }
}

 template<typename T>
 __device__ void backpolate_bicubic(T *out, T error, T idx, T idy, int width, int height)
{
    const int idy_f = floor(idy);

    T buff_y[4] = {0,0,0,0};
    backpolate_cubic<T>(buff_y, error, idy - idy_f + 1, 4, true);

    for (int dy = -1; dy < 3; ++dy)
    {
        const int c_idx_y = idy_f + dy;
        if (c_idx_y >= 0 && c_idx_y < height)
            backpolate_cubic<T>(&(out[width*c_idx_y]), buff_y[dy+1], idx, width);
    }
}


template<typename T>
__global__ void rotate_kernel_grad(
    const typename optox::DTensor<T, 1>::ConstRef angles,
    const typename optox::DTensor<T, 5>::ConstRef grad_out,
    typename optox::DTensor<T, 4>::Ref grad_x)
{
    const int idx_in = threadIdx.x;
    const int idy_in = threadIdx.y;
    const int idz_in = threadIdx.z + blockIdx.z * blockDim.z;

    const int ids = idz_in % grad_x.size_[0];
    const int idf = idz_in / grad_x.size_[0];

    const int kernel_height = grad_x.size_[2];
    const int kernel_width = grad_x.size_[3];

    extern __shared__ __align__(sizeof(T)) unsigned char s_buffer[];
    T *grad_x_shared = reinterpret_cast<T*>(s_buffer);

    // load data into shared memory (used to interpolate)
    grad_x_shared[idx_in+kernel_width*idy_in] = 0;

    __syncthreads();

    const int center_x = kernel_width / 2;
    const int center_y = kernel_height / 2;
    // compute the centralized coordinates
    const int idx_in_c = idx_in - center_x;
    const int idy_in_c = -idy_in + center_y;

    for (int angle_idx = 0; angle_idx < angles.size_[0]; ++angle_idx)
    {
        const T theta = angles(angle_idx);
        // rotate the coordinate into the source image (-theta)
        const T cos_theta = cos(theta);
        const T sin_theta = sin(theta);
        const T idx_out_c = cos_theta * idx_in_c + sin_theta * idy_in_c;
        const T idy_out_c = -sin_theta * idx_in_c + cos_theta * idy_in_c;

        // transfer them back into the input image (in and out have same size)
        const T idx_out = idx_out_c + center_x;
        const T idy_out = -idy_out_c + center_y;

        backpolate_bicubic<T>(grad_x_shared, grad_out(ids, angle_idx, idf, idy_in, idx_in),
            idx_out, idy_out, kernel_width, kernel_height);
    }

    __syncthreads();

    // write the result into global memory
    grad_x(ids, idf, idy_in, idx_in) = grad_x_shared[idx_in+kernel_width*idy_in];
}

template<typename T>
void optox::RotOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_y = this->template getInput<T, 5>(0, inputs);
    auto angles = this->template getInput<T, 1>(1, inputs);
    auto grad_x = this->template getOutput<T, 4>(0, outputs);

    if (angles->size()[0] != grad_y->size()[1])
        THROW_OPTOXEXCEPTION("RotOperator: angles and grad output require same size");

    dim3 block_size(grad_x->size()[3],
                    grad_x->size()[2]);
    dim3 grid_size(1, 1, grad_x->size()[0]*grad_x->size()[1]);

    unsigned int smem_size = grad_x->size()[2] * grad_x->size()[3] * sizeof(T);
    rotate_kernel_grad<T><<<grid_size, block_size, smem_size, this->stream_>>>(
        *angles, *grad_y, *grad_x);
    OPTOX_CUDA_CHECK;
}


#define REGISTER_OP(T) \
    template class optox::RotOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
