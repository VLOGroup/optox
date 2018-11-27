#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/register_types.h"
#include "pad2d.h"
#include "definitions.h"

inline __device__ int symXpixel(int x, int width, bool skip_edge = false)
{
  int src_x = x;
  // skip the border pixel to get iid noise
  if (x < 0)
    src_x = abs(x) - 1 + (skip_edge ? 1 : 0);
  else if (x >= width)
    src_x = 2 * width - 1 - x - (skip_edge ? 1 : 0);
  return src_x;
}

inline __device__ int symYpixel(int y, int height, bool skip_edge = false)
{
  int src_y = y;
  // skip the border pixel to get iid noise
  if (y < 0)
    src_y = abs(y) - 1 + (skip_edge ? 1 : 0);
  else if (y >= height)
    src_y = 2 * height - 1 - y - (skip_edge ? 1 : 0);
  return src_y;
}

template<typename T>
__global__ void Pad2dReplicateKernel(const typename Tensor3<T>::ConstTensor in,
                                     typename Tensor3<T>::Tensor out,
                                     int pad_x, int pad_y)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < out.dimensions()[2] && y < out.dimensions()[1] && z < out.dimensions()[0])
  {
    int x_in = max(0, min(x - pad_x, int(in.dimensions()[2]) - 1));
    int y_in = max(0, min(y - pad_y, int(in.dimensions()[1]) - 1));

    out(z, y, x) = in(z, y_in, x_in);
  }
}

template<typename T>
__global__ void Pad2dSymmetricKernel(const typename Tensor3<T>::ConstTensor in,
                                     typename Tensor3<T>::Tensor out,
                                     int pad_x, int pad_y)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < out.dimensions()[2] && y < out.dimensions()[1] && z < out.dimensions()[0])
  {
    const int x_in = symXpixel(x - pad_x, int(in.dimensions()[2]));
    const int y_in = symYpixel(y - pad_y, int(in.dimensions()[1]));

    out(z, y, x) = in(z, y_in, x_in);
  }
}

template<typename T>
__global__ void Pad2dTransposeReplicateKernel(const typename Tensor3<T>::ConstTensor in,
                                              typename Tensor3<T>::Tensor out,
                                              int crop_x, int crop_y)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < out.dimensions()[2] && y < out.dimensions()[1] && z < out.dimensions()[0])
  {

    bool value_inside = ((x >= 1) && (x < out.dimensions()[2] - 1) && (y >= 1)
        && (y < out.dimensions()[1] - 1));

    T value = static_cast<T>(0);

    if (value_inside)
      value = in(z, y + crop_y, x + crop_x);
    else
    {
      if ((x == 0) && (y == 0))  // upper left corner
      {
        for (int dx = -crop_x; dx <= 0; dx++)
          for (int dy = -crop_y; dy <= 0; dy++)
            value += in(z, crop_y + dy, crop_x + dx);
      }
      else if ((x == 0) && (y == out.dimensions()[1] - 1))  // lower left corner
      {
        for (int dx = -crop_x; dx <= 0; dx++)
          for (int dy = 0; dy <= crop_y; dy++)
            value += in(z, out.dimensions()[1] - 1 + crop_y + dy, crop_x + dx);
      }
      else if ((x == out.dimensions()[2] - 1) && (y == out.dimensions()[1] - 1))  // lower right corner
      {
        for (int dx = 0; dx <= crop_x; dx++)
          for (int dy = 0; dy <= crop_y; dy++)
            value += in(z, out.dimensions()[1] - 1 + crop_y + dy,
                                 out.dimensions()[2] - 1 + crop_x + dx);
      }
      else if ((x == out.dimensions()[2] - 1) && (y == 0))  // upper right corner
      {
        for (int dx = 0; dx <= crop_x; dx++)
          for (int dy = -crop_y; dy <= 0; dy++)
            value += in(z, crop_y + dy, out.dimensions()[2] - 1 + crop_x + dx);
      }
      else if (y == 0)  // upper
      {
        for (int dy = -crop_y; dy <= 0; dy++)
          value += in(z, crop_y + dy, x + crop_x);
      }
      else if (y == out.dimensions()[1] - 1)  // lower
      {
        for (int dy = 0; dy <= crop_y; dy++)
          value += in(z, out.dimensions()[1] - 1 + crop_y + dy, x + crop_x);
      }
      else if (x == 0)  // left
      {
        for (int dx = -crop_x; dx <= 0; dx++)
          value += in(z, y + crop_y, crop_x + dx);
      }
      else if (x == out.dimensions()[2] - 1)  // right
      {
        for (int dx = 0; dx <= crop_x; dx++)
          value += in(z, y + crop_y, out.dimensions()[2] - 1 + crop_x + dx);
      }
    }

    out(z, y, x) = value;
  }
}

template<typename T>
__global__ void Pad2dTransposeSymmetricKernel(const typename Tensor3<T>::ConstTensor in,
                                              typename Tensor3<T>::Tensor out,
                                              int crop_x, int crop_y)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int z = threadIdx.z + blockIdx.z * blockDim.z;

  if (x < out.dimensions()[2] && y < out.dimensions()[1] && z < out.dimensions()[0])
  {
    bool value_inside = ((x >= crop_x) && (x < out.dimensions()[2] - crop_x)
        && (y >= crop_y) && (y < out.dimensions()[1] - crop_y));

    T value = static_cast<T>(0);

    value = in(z, y + crop_y, x + crop_x);
    if (!value_inside)
    {
      if ((x < crop_x) && (y < crop_y))  // upper left corners
      {
        int dx = x + 1;
        int dy = y + 1;
        value +=  in(z, crop_y - dy, crop_x - dx);
      }
      if ((x < crop_x) && (y >= out.dimensions()[1] - crop_y))  // lower left corner
      {
        int dx = x + 1;
        int dy = 2 * out.dimensions()[1] - 1 - y;
        value +=  in(z, crop_y + dy, crop_x - dx);
      }
      if ((x >= out.dimensions()[2] - crop_x) && (y >= out.dimensions()[1] - crop_y))  // lower right corner
      {
        int dx = 2 * out.dimensions()[2] - 1 - x;
        int dy = 2 * out.dimensions()[1] - 1 - y;
        value +=  in(z, crop_y + dy, crop_x + dx);
      }
      if ((x >= out.dimensions()[2] - crop_x) && (y < crop_y))  // upper right corner
      {
        int dx = 2 * out.dimensions()[2] - 1 - x;
        int dy = y + 1;
        value += + in(z, crop_y - dy, crop_x + dx);
      }
      if (y < crop_y)  // upper
      {
        int dy = y + 1;
        value += in(z, crop_y - dy, crop_x + x);
      }
      if (y >= out.dimensions()[1] - crop_y)  // upper
      {
        int dy = 2 * out.dimensions()[1] - 1 - y;
        value += in(z, crop_y + dy, crop_x + x);
      }
      if (x < crop_x)  // left
      {
        int dx = x + 1;
        value += in(z, crop_y + y, crop_x - dx);
      }
      if (x >= out.dimensions()[2] - crop_x)  // right
      {
        int dx = 2 * out.dimensions()[2] - 1 - x;
        value += in(z, crop_y + y, crop_x + dx);
      }
    }
    out(z, y, x) = value;
  }
}

template<typename T>
void Pad2dKernelLauncher(const tensorflow::Tensor * in,
                               tensorflow::Tensor * out,
                         const int pad, const tficg::borderMode_t mode)
{
  auto kd_in = in->flat_inner_dims<T,3>();
  auto kd_out = out->flat_inner_dims<T,3>();

  dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(
      divUp(kd_out.dimensions()[2], dimBlock.x),
      divUp(kd_out.dimensions()[1], dimBlock.y),
      divUp(kd_out.dimensions()[0], dimBlock.z));

  switch(mode)
  {
    case tficg::BORDER_MODE_REPLICATE:
      Pad2dReplicateKernel<T><<<dimGrid, dimBlock>>>(kd_in, kd_out, pad, pad);
      break;
    case tficg::BORDER_MODE_SYMMETRIC:
      Pad2dSymmetricKernel<T><<<dimGrid, dimBlock>>>(kd_in, kd_out, pad, pad);
      break;
  }
}

template <typename T>
void Pad2dTransposeKernelLauncher(const tensorflow::Tensor * in,
                               tensorflow::Tensor * out,
                               const int pad, const tficg::borderMode_t mode)
{
  auto kd_in = in->flat_inner_dims<T,3>();
  auto kd_out = out->flat_inner_dims<T,3>();

  dim3 dimBlock(ICGVN_BLOCK_SIZE_3D_X, ICGVN_BLOCK_SIZE_3D_Y, ICGVN_BLOCK_SIZE_3D_Z);
  dim3 dimGrid(
      divUp(kd_out.dimensions()[2], dimBlock.x),
      divUp(kd_out.dimensions()[1], dimBlock.y),
      divUp(kd_out.dimensions()[0], dimBlock.z));

  switch(mode)
  {
    case tficg::BORDER_MODE_REPLICATE:
      Pad2dTransposeReplicateKernel<T><<<dimGrid, dimBlock>>>(kd_in, kd_out, pad, pad);
      break;
    case tficg::BORDER_MODE_SYMMETRIC:
      Pad2dTransposeSymmetricKernel<T><<<dimGrid, dimBlock>>>(kd_in, kd_out, pad, pad);
      break;
  }
}

#define REGISTER_KERNEL_LAUNCHER(T) \
    template void Pad2dKernelLauncher<T>(const tensorflow::Tensor * in, \
                                         tensorflow::Tensor * out, \
                                         const int pad, \
                                         const tficg::borderMode_t mode); \
    template void Pad2dTransposeKernelLauncher<T>(const tensorflow::Tensor * in, \
                                         tensorflow::Tensor * out, \
                                         const int pad, \
                                         const tficg::borderMode_t mode);

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_KERNEL_LAUNCHER);

#undef REGISTER_KERNEL_LAUNCHER

#endif
