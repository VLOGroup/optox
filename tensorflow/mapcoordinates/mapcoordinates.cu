/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/contrib/icg/ops/mapcoordinates.h"
#include "mapcoordinates.h"
#include "atomicAddFloat64.h"

// tensorflow::CudaAtomicAdd => requires compute capability >= 3.0
// if using nvcc directly specify a compute capability >= 3.0 with the following flags -gencode arch=compute_30,code=sm_30
//#include "tensorflow/core/util/cuda_kernel_helper.h"

// use CUDA Specialized GPU functions that provide STL similar function on GPU
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include <cuda.h>

// using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


 template<typename T>
 __device__ T interpolate_cubic(volatile T *in, T du,bool difference2Point = false)
 {

   // get the input values
   T i_f   = in[1];
   T i_c   = in[2];
   T i_f_1, i_c_1;
   int delta;
   if (difference2Point){
    i_f_1 = i_f;
    i_c_1 = i_c;
    delta = 1;
   }else{
    i_f_1 = in[0];
    i_c_1 = in[3];
    delta = 2;
  }
  // Explanations: 
  // t value to be interpolated at, t_f floored 
  //  # // from wikipedia cubic_hermite_spline 
  //  # // p(t)  = (2t³-3t²+1)p0 + (t³-2t² +t)m0 + (-2t³+3t²)p1 + (t³-t²)m1
  //  # => reordering for efficiency of calculations
  //  #   p(t) = t³*(2p0 - 2p1  + m0 + m1) + t²*(-3p0 -3p1 +2m0 - m1) + t*m0 + 1*p0
  //      
  //  # derivatives (2 sided difference, needs 4 points):
  //  # m1 = p’(t_f) =  ( p(t_f+1) -  p(t_f-1) ) / 2 =  (p(t_c)   - p(t_f-1)) /2
  //  # m0 = p'(t_c) =  ( p(t_c+1) -  p(t_c-1) ) / 2 =  (p(t_c+1) - p(t_f)  ) /2
  //    
  //  # derivatives (1 sided as in scipy ndimage):
  //  # m1 = p’(t_f) =   ( p(t_f+1) -  p(t_f) ) / 2 =  (p(t_c) - p(t_f)) /1
  //  # m0 = p'(t_c) =  -( p(t_c-1) -  p(t_c) ) / 2 =  (p(t_c) - p(t_f)) /1 
  //   => for scipy compatibility use in[0] = in[1] & in[2] = in[3]

   // determine the coefficients
   const T p_f = i_f;
   const T p_prime_f = (i_c - i_f_1) / delta;
   const T p_c = i_c;
   const T p_prime_c = (i_c_1 - i_f) / delta;

   const T a =  2*p_f - 2*p_c +   p_prime_f + p_prime_c;
   const T b = -3*p_f + 3*p_c - 2*p_prime_f - p_prime_c;
   const T c = p_prime_f;
   const T d = p_f;

   T out = du*(du*(du*a+b)+c) + d;
  //  d + u*c + u²b + u³ a

   return out;
 }


template <typename T>
__global__ void MapcoordinatesGPUBicubic(const typename Tensor4<T>::ConstTensor  img ,
                  const typename Tensor4<T>::ConstTensor coords,
                  typename Tensor4<T>::Tensor out, int interp_type){  
  // ATTENTION: Tensor object must not be passed via reference into CUDA Kernel!
  // => Tensor DATA is on GPU but extra information like .dimensions() is on CPU heap => invalid pointers
  // => pass by value copies this extra informaiton from CPU to GPU 
  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
  const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
  const unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

  // Cuda only has 3 dimensions => split idz into 2 dimensions 
  const unsigned int idn =  idz / wdc;
  const unsigned int idc =  idz % wdc;

  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){      
    const T x = coords(idn,1, idy, idx);
    const T y = coords(idn,0, idy, idx);
    const int x1 = floor( (float) x);
    const int y1 = floor( (float) y);

    int neglim = 0;
    int poslim = 1;
    bool difference2Point = true;
    if (interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS){
      neglim = -1;
      poslim = 2;
      difference2Point = false;
    }
    
    T buff_y[4] = {0,0,0,0};

    for (int dy = neglim; dy <= poslim; ++dy)
    {
      const int c_idy = y1 + dy;
      T buff_vals[4] = {0,0,0,0};// = {0,0,0,0};
      if ((c_idy >= 0) && (c_idy < wdy)){
        for (int dx  = neglim; dx <= poslim; ++dx){
          const int c_idx = x1 +dx;
          if ( (c_idx>=0) && (c_idx<wdx) ){
            buff_vals[1+dx] = img(idn,c_idy,c_idx,idc);
          }
          else
            buff_vals[1+dx] = 0;
        }
        buff_y[1+dy] = interpolate_cubic<T>( buff_vals , x - x1, difference2Point);
      }
      else
        buff_y[1+dy] = 0;
    }

    T c_out = interpolate_cubic<T>(buff_y, y - y1, difference2Point);
    if (((y1+neglim)>= 0) && ((y1+poslim) < wdy) && ((x1+neglim) >= 0) && ((x1+poslim) < wdx))
      out(idn,idy,idx,idc) = c_out;
    else
      out(idn,idy,idx,idc) = 0;
  }
}


template <typename T>
__global__ void MapcoordinatesGPU(const typename Tensor4<T>::ConstTensor  img ,
                  const typename Tensor4<T>::ConstTensor coords,
                  typename Tensor4<T>::Tensor out, int interp_type){  
  // ATTENTION: Tensor object must not passed via reference into CUDA Kernel!
  // => Tensor DATA is on GPU but extra information like .dimensions() is on CPU heap => invalid pointers
  // => pass by value copies this extra informaiton from CPU to GPU 
  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
  const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
  const unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

  // Cuda only has 3 dimensions => split idz into 2 dimensions 
  const unsigned int idn =  idz / wdc;
  const unsigned int idc =  idz % wdc;

  // printf(" kernel %d, %d, %d, %d,  idn:%d,idy:%d,idx:%d,idc:%d,idz:%d\n", wdn,wdy ,wdx,wdc, idn,idy,idx,idc,idz);
  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){
    T x = coords(idn,1, idy, idx);
    T y = coords(idn,0, idy, idx);
    int x1 = floor( (float) x);
    int y1 = floor( (float) y);
    int x2 = x1+1; //ceil = floor + 1
    int y2 = y1+1;
    T values_at_top_left   = 0;
    T values_at_top_right  = 0;
    T values_at_bot_left   = 0;
    T values_at_bot_right  = 0;

    if ((y1 >= 0) && (y1 < wdy) && (x1 >= 0) && (x1 < wdx))
      values_at_top_left  = img(idn,y1,x1,idc);

    if ((y1 >= 0) && (y1 < wdy) && (x2 >= 0) && (x2 < wdx))
      values_at_top_right = img(idn,y1,x2,idc);

    if ((y2 >= 0) && (y2 < wdy) && (x1 >= 0) && (x1 < wdx))
      values_at_bot_left  = img(idn,y2,x1,idc);

    if ((y2 >= 0) && (y2 < wdy) && (x2 >= 0) && (x2 < wdx))
      values_at_bot_right = img(idn,y2,x2,idc);

    T horizontal_interpolated_top = values_at_top_left  + (coords(idn,1,idy,idx) - x1 ) * (values_at_top_right-values_at_top_left);
    T horizontal_interpolated_bot = values_at_bot_left  + (coords(idn,1,idy,idx) - x1 ) * (values_at_bot_right-values_at_bot_left);
    T interpolated_result = horizontal_interpolated_top + (coords(idn,0,idy,idx) - y1 ) * ( horizontal_interpolated_bot - horizontal_interpolated_top);

    if ((y1 >= 0) && (y2 < wdy) && (x1 >= 0) && (x2 < wdx))
      out(idn,idy,idx,idc) = interpolated_result;
    else
      out(idn,idy,idx,idc) = 0;
  }
  // else
    // printf(" kernel not ex %d, %d, %d, %d,  idn:%d,idy:%d,idx:%d,idc:%d,idz:%d\n", wdn,wdy ,wdx,wdc, idn,idy,idx,idc,idz);
}


// Implementing the CPU version of the functor here
template <typename T>
struct MapcoordinatesFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor &coords,
                  typename Tensor4<T>::Tensor &out, tficg::interp_type_t interp_type)
                    {
    const GPUDevice d = context->eigen_device<GPUDevice>();
    const int wdn = img.dimensions()[0];
    const int wdy = img.dimensions()[1];
    const int wdx = img.dimensions()[2];
    const int wdc = img.dimensions()[3];    

    // printf("   IMG (NHWC): %d, %d, %d, %d\n", wdn,wdy ,wdx,wdc);
    // printf("coords (N2HW): %ld, %ld, %ld, %ld\n", coords.dimensions()[0],coords.dimensions()[1] ,coords.dimensions()[2], coords.dimensions()[3]);
    // printf("   out (NHWC): %ld, %ld, %ld, %ld\n",    out.dimensions()[0],   out.dimensions()[1] ,   out.dimensions()[2],    out.dimensions()[3]);


    // unsigned int N = wdn*wdy*wdx*wdc;
    // // initialize gradients with 0s, watch out to tell thrust that the pointer is on the GPU device!
    thrust::fill(thrust::device_ptr<T>(out.data()), 
                 thrust::device_ptr<T>(out.data()+out.size()),
                  T(0));

    dim3 block_size(32,32,1);
    int gdx = (std::ceil( wdx /static_cast<float>(block_size.x )));
    int gdy = (std::ceil( wdy /static_cast<float>(block_size.y ))); 
    int gdz = (std::ceil( wdn*wdc /static_cast<float>(block_size.z ))); 
    dim3 grid_size( gdx,gdy,gdz); 

    // printf("   gdx: %d, gdy: %d, gdz:%d \n", gdx,gdy ,gdz);

    // MapcoordinatesGPU<T><<<grid_size,block_size>>>(img ,coords, out,  interp_type);
    if (interp_type == tficg::INTERP_TYPE_BILINEAR)
      MapcoordinatesGPU<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords, out,  interp_type);
    else if (interp_type == tficg::INTERP_TYPE_BICUBIC_2POINTS or interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS)
      MapcoordinatesGPUBicubic<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords, out,  interp_type);
    else
      printf("\n !!!ERROR UNKNOWN FILTER INTERPOLATION TYPE !!!\n ");
    // cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("\nCUDA Error: %s\n", cudaGetErrorString(err));

    // printf(" kernel calll finished\n");
    // std::cout << cudaerr << std::endl;
    // MapcoordinatesGPU<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords, out,  interp_type);
  }
};


// template struct MapcoordinatesFunctor<GPUDevice, float>
// template struct MapcoordinatesFunctor<GPUDevice, int>

#define REGISTER_GPU_FUNCTOR(T) \
template struct MapcoordinatesFunctor<GPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR







// template <typename T>
// __global__ void Mapcoordinates3DGPU(const typename tensorflow::TTypes<T,5>::ConstTensor  img ,
//                   const typename tensorflow::TTypes<T,5>::ConstTensor coords,
//                   typename tensorflow::TTypes<T,5>::Tensor out,
//                   bool repetetive_padding_in_z) { 
//   // chart inspired from wikipedia: https://en.wikipedia.org/wiki/Bilinear_interpolation
//   // but Top & Bottom flipped to fit plotting standards
//   //      Q112       Rx12         Q212
//   //          *-------*-----------*
//   //         /       /|          /|
//   //        /       / |         / |
//   //   Q111/     Rx11 |    Q211/  |     
//   // y1   *-------*-----------*   |     
//   //      |       |   *Pxy2   |   |     
//   //      |       |  /|       |   |     
//   //      |       | x *-------|---*    z2 
//   //      |       |/ \        |  /      
//   // y    |   Pxy1* / Vxyz    | /       
//   //      |       |/          |/        
//   // y2   *-------*-----------*    z1     
//   //     Q121    Rx21      Q221
//   //     (x1)    (x)       (x2)
//   //
//   //  Numbers are xyz as in wikipedia => Q121 = Q(y2,x1,z1) (Tf = NHWC)
//   //
//   // f(R_11) = f(Q111) + (x-x1) * ( f(Q211) - f(Q111) )  //interp  x1..x..x2 @ y1,z1
//   // f(R_21) = f(Q121) + (x-x1) * ( f(Q221) - f(Q121) )  //interp  x1..x..x2 @ y2,z1
//   // f(P__1) = f(Rx11) + (y-y1) * ( f(Rx21) - f(Rx11) )  //interp  y1..y..y2 @ x,z1
//   // f(R_12) = f(Q112) + (x-x1) * ( f(Q212) - f(Q112) )  //interp  x1..x..x2 @ y1,z1
//   // f(R_22) = f(Q122) + (x-x1) * ( f(Q222) - f(Q122) )  //interp  x1..x..x2 @ y1,z2
//   // f(P__2) = f(Rx12) + (y-y1) * ( f(Rx22) - f(Rx12) )  //interp  y1..y..y2 @ x,z2
//   // f(Vxyz) = f(Pxy1) + (z-z1) * ( f(Pxy2) - f(Pxy1) )  //interp  z1..z..z2 @ x,y

//   const int wdn = img.dimensions()[0];
//   const int wdy = img.dimensions()[1];
//   const int wdx = img.dimensions()[2];
//   const int wdc = img.dimensions()[3];
//   const int wdz = img.dimensions()[4];

//   const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
//   const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
//         unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

//   // Cuda only has 3 dimensions => split idz into the other dimensions 
//   const unsigned int idn =  idz / (wdc*wdz);
//   const unsigned int tmp =  idz % (wdc*wdz);
//   const unsigned int idc =  tmp / wdz;
//                      idz =  tmp % wdz;

//   // printf(" kernel %d, %d, %d, %d,  idn:%d,idy:%d,idx:%d,idc:%d,idz:%d\n", wdn,wdy ,wdx,wdc, idn,idy,idx,idc,idz);
//   if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) && (idz<wdz) ) {
    
//   // coords (idn,0..2,idy,idx,idz)        // N3HWD 0=y,1=x,2=z
//   // img (idn,idy,idx,idc,idz)            // NHWCD
//     const T x = coords(idn,1, idy, idx,idz);
//     const T y = coords(idn,0, idy, idx,idz);
//     T z = coords(idn,2, idy, idx,idz);

//     if (repetetive_padding_in_z){
//       // for some strange reason z %= wdz doesn't work...
//       while (z>wdz) z -= wdz;
//       while (z<0)   z += wdz;
//     }
//     const int x1 = std::floor(x);
//     const int y1 = std::floor(y);
//     const int z1 = std::floor(z);
//     const int x2 = x1+1; //ceil = floor + 1
//     const int y2 = y1+1;
//     const int z2 = z1+1;

//     T fQ111 = 0;
//     T fQ121 = 0;
//     T fQ211 = 0;
//     T fQ221 = 0;
//     T fQ112 = 0;
//     T fQ122 = 0;
//     T fQ212 = 0;
//     T fQ222 = 0;

//     if( (z1 >= 0) && (z1 < wdz) ){
//       if ( (y1 >= 0) && (y1 < wdy) ){
//         if( (x1 >= 0) && (x1 < wdx) ){
//           fQ111 = img(idn,y1,x1,idc,z1);
//         }
//         if ( (x2 >= 0) && (x2 < wdx)){
//           fQ211 = img(idn,y1,x2,idc,z1);  
//         }
//       }
//       if ( (y2 >= 0) && (y2 < wdy)){
//         if ((x1 >= 0) && (x1 < wdx) ){
//           fQ121 = img(idn,y2,x1,idc,z1);
//         }
//         if ( (x2 >= 0) && (x2 < wdx)){
//           fQ221 = img(idn,y2,x2,idc,z1);
//         }
//       }
//     }
//     if ( (z2 >= 0) && (z2 < wdz) ){
//       if ( (y1 >= 0) && (y1 < wdy) ){
//         if( (x1 >= 0) && (x1 < wdx) ){
//           fQ112 = img(idn,y1,x1,idc,z2);
//         }
//         if ( (x2 >= 0) && (x2 < wdx)){
//           fQ122 = img(idn,y2,x1,idc,z2);
//         }
//       }
//       if ( (y2 >= 0) && (y2 < wdy) ){
//         if ((x1 >= 0) && (x1 < wdx) ){
//           fQ212 = img(idn,y1,x2,idc,z2);
//         }
//         if ((x2 >= 0) && (x2 < wdx) ){
//           fQ222 = img(idn,y2,x2,idc,z2);
//         }
//       }
//     }
//     //  1.) 2D inpterpolate @ z = z1 
//     const T fRx11 = fQ111 + (x-x1) * ( fQ211 - fQ111);  //interp  x1..x..x2 @ y1,z1
//     const T fRx21 = fQ121 + (x-x1) * ( fQ221 - fQ121);  //interp  x1..x..x2 @ y2,z1
//     const T fPxy1 = fRx11 + (y-y1) * ( fRx21 - fRx11);  //interp  y1..y..y2 @ x,z1
//     //  2.) 2D inpterpolate @ z = z2
//     const T fRx12 = fQ112 + (x-x1) * ( fQ212 - fQ112);  //interp  x1..x..x2 @ y1,z2
//     const T fRx22 = fQ122 + (x-x1) * ( fQ222 - fQ122);  //interp  x1..x..x2 @ y2,z2
//     const T fPxy2 = fRx12 + (y-y1) * ( fRx22 - fRx12);  //interp  y1..y..y2 @ x,z2
//     //  3.) interpoltate between the two points.. 
//     const T fVxyz = fPxy1 + (z-z1) * ( fPxy2 - fPxy1);  //interp  z1..z..z2 @ x,y

//     if ((y1 >= 0) && (y2 < wdy) && (x1 >= 0) && (x2 < wdx))
//       if (repetetive_padding_in_z)
//         out(idn,idy,idx,idc,idz) = fVxyz;
//       else if  ((z1 >= 0) && (z2 < wdz))
//         out(idn,idy,idx,idc,idz) = fVxyz;
//       else 
//         out(idn,idy,idx,idc,idz) = 0;
//     else
//       out(idn,idy,idx,idc,idz) = 0;
//   }
// }


// // specify the Functor once in general form in the header for CPU & GPU version
// template<typename Device, typename T>
// struct Mapcoordinates3DFunctor{
//   void operator()(tensorflow::OpKernelContext* context, 
//                   const typename tensorflow::TTypes<T,5>::ConstTensor  &img ,
//                   const typename tensorflow::TTypes<T,5>::ConstTensor &coords,
//                   typename tensorflow::TTypes<T,5>::Tensor &out);
// };

// // Implementing the GPU version of the functor here
// template <typename T>
// struct Mapcoordinates3DFunctor<GPUDevice, T> {
//   void operator()(tensorflow::OpKernelContext* context, 
//                   const typename tensorflow::TTypes<T,5>::ConstTensor  &img ,
//                   const typename tensorflow::TTypes<T,5>::ConstTensor &coords,
//                   typename tensorflow::TTypes<T,5>::Tensor &out)
//                     {
   
//     dim3 block_size(32,32,1);
//     dim3 grid_size( 1,1,1); 
//     Mapcoordinates3DGPU<T><<<grid_size,block_size>>>(img ,coords, out,  true);
//   }
// };

// #define REGISTER_GPU_FUNCTOR(T) \
// template struct Mapcoordinates3DFunctor<GPUDevice, T>;
// // for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
// TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU_FUNCTOR);
// // TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_FUNCTOR);
// #undef REGISTER_GPU_FUNCTOR






// ###################################################################################################################################################
// GRADIENT SECTION
// ###################################################################################################################################################



/**
 * perform bilinear interpolation on the input image i_in given the
 * index (idx,idy)
 */
 template<typename T>
 __device__ void backpolate_cubic(T *out, T gradIn, T dt, bool difference2PointNot4Point = false){
   const T u = dt;
   const T uu = u*u;
   const T uuu = uu*u;
   
   if (difference2PointNot4Point){
     out[0] = 0;
     out[1] = (1-u)   * gradIn;
     out[2] = u * gradIn;
     out[3] = 0;
     
   }else{
    // determine the coefficients
     T d_out_d_p_f_1 = -uuu/2 + uu - u/2;
     T d_out_d_p_f = (3*uuu)/2 - (5*uu)/2 + 1;
     T d_out_d_p_c = -(3*uuu)/2 + 2*uu + u/2;
     T d_out_d_p_c_1 = uuu/2 - uu/2;
     
     out[0] = d_out_d_p_f_1 * gradIn;
     out[1] = d_out_d_p_f   * gradIn;
     out[2] = d_out_d_p_c   * gradIn;
     out[3] = d_out_d_p_c_1 * gradIn;
   }
 }



template <typename T>
__global__ void MapcoordinatesGradientsGPUBicubic(
                  const typename Tensor4<T>::ConstTensor  img ,
                  const typename Tensor4<T>::ConstTensor  coords ,
                  const typename Tensor4<T>::ConstTensor  gradIn ,
                  typename Tensor4<T>::Tensor gradImg,
                  typename Tensor4<T>::Tensor gradCoords,
                  int interp_type){  
  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
  const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
  const unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

  // Cuda only has 3 dimensions => split idz into 2 dimensions 
  const unsigned int idn =  idz / wdc;
  const unsigned int idc =  idz % wdc;

  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){
    T currGrad = gradIn(idn,idy,idx,idc);

    const T x = coords(idn,1, idy, idx);
    const T y = coords(idn,0, idy, idx);
    const int x1 = floor( (float) x);
    const int y1 = floor( (float) y);

    int neglim = 0;
    int poslim = 1;
    bool difference2Point = true;
    if (interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS){
      neglim = -1;
      poslim = 2;
      difference2Point = false;
    }


    T buff_y[4] = {0,0,0,0};
    // calculate where the gradient came from
    backpolate_cubic<T>(buff_y, currGrad, y - y1,difference2Point);

    // use the exact same for loop to spread the gradient to the correct pixels
    for (int dy = neglim; dy <= poslim; ++dy)
    {
      const int c_idy = y1 + dy;
      T buff_vals[4] = {0,0,0,0};
      if ((c_idy >= 0) && (c_idy < wdy)){

        // calculate where the gradient came from
        currGrad = buff_y[1+dy];
        backpolate_cubic<T>(buff_vals, currGrad, x - x1 ,difference2Point);

        for (int dx  = neglim; dx <= poslim; ++dx){
          const int c_idx = x1 + dx;
          if ( (c_idx>=0) && (c_idx<wdx) &&
            ((y1+neglim)>= 0) && ((y1+poslim) < wdy) && ((x1+neglim) >= 0) && ((x1+poslim) < wdx) ) {
            // Write the value back to the 
            //tensorflow::CudaAtomicAdd(&gradImg(idn,c_idy,c_idx,idc), buff_vals[1+dx]);
            atomicAdd(&gradImg(idn,c_idy,c_idx,idc), buff_vals[1+dx]);
          }
        }
      }
    }
  }
};


template <typename T>
__global__ void MapcoordinatesGradientsGPU(
                  const typename Tensor4<T>::ConstTensor  img ,
                  const typename Tensor4<T>::ConstTensor  coords ,
                  const typename Tensor4<T>::ConstTensor  gradIn ,
                  typename Tensor4<T>::Tensor gradImg,
                  typename Tensor4<T>::Tensor gradCoords,
                  int interp_type){  
  // ATTENTION: Tensor object must not passed via reference into CUDA Kernel!
  // => Tensor DATA is on GPU but extra information like .dimensions() is on CPU heap => invalid pointers
  // => pass by value copies this extra informaiton from CPU to GPU 


// chart inspired from wikipedia: https://en.wikipedia.org/wiki/Bilinear_interpolation
// but Top & Bottom flipped to fit plotting standards
// 
//     Q11      R1      Q21         Q11(x1,y1)  values_at_top_left
// y1   *-------*---------*         Q21(x2,y1)  values_at_top_right
//      |       |         |         Q12(x1,y2)  values_at_bot_left
//      |       |         |         Q22(x2,y2)  values_at_bot_right
// y    |       *P        |         R1(x,y1)    horizontal_interpolated_top
//      |       |         |         R2(x,y2)    horizontal_interpolated_bot
// y2   *-------*---------*         P(x,y)      interpolated_result
//     Q12      R2       Q22
//     (x1)    (x)       (x2)
//
// f(R1) = f(Q11) + (x-x1) * ( f(Q21) - f(Q11) )
// f(R2) = f(Q12) + (x-x1) * ( f(Q22) - f(Q12) )
//  f(P) = f(R1)  + (y-y1) * ( f(R1)  - f(R2)  )
//
// Gradients (for one pixel, need to be summed for all)
//   Gradients wrt. the pixel values (gradImg)
//   @(x1,y1): dL/dQ11 = dl/df(R(x,y)) * (y2 - y ) * (x2 - x ) 
//   @(x2,y1): dL/dQ21 = dl/df(R(x,y)) * (y2 - y ) * (x  - x1) 
//   @(x1,y2): dL/dQ12 = dl/df(R(x,y)) * (y  - y1) * (x2 - x ) 
//   @(x2,y2): dL/dQ22 = dl/df(R(x,y)) * (y  - y1) * (x  - x1)   
//   Gradients wrt. the coordinates (gradCoords)
//   @(x,y)  : dL/dx   = dl/df(R(x,y)) * ( (y2-y)*(f(Q21)-f(Q11)) + (y-y1)*(f(Q22)-f(Q12)) )
//   @(x,y)  : dL/dy   = dl/df(R(x,y)) * ( f(R2)-f(R1) )
//
//   gradIn(x,y) := dl/df(R(x,y)) 

  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
  const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
  const unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

  // Cuda only has 3 dimensions => split idz into 2 dimensions 
  const unsigned int idn =  idz / wdc;
  const unsigned int idc =  idz % wdc;

  // printf(" kernel %d, %d, %d, %d,  idn:%d,idy:%d,idx:%d,idc:%d,idz:%d\n", wdn,wdy ,wdx,wdc, idn,idy,idx,idc,idz);
  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){
    const T x = coords(idn,1, idy, idx);
    const T y = coords(idn,0, idy, idx);
    const int x1 = floor( (float) x);
    const int y1 = floor( (float) y);
    const int x2 = x1+1; //ceil = floor + 1
    const int y2 = y1+1;

    T values_at_top_left   = 0;
    T values_at_top_right  = 0;
    T values_at_bot_left   = 0;
    T values_at_bot_right  = 0;

    if ((y1 >= 0) && (y1 < wdy) && (x1 >= 0) && (x1 < wdx))
      values_at_top_left  = img(idn,y1,x1,idc);

    if ((y1 >= 0) && (y1 < wdy) && (x2 >= 0) && (x2 < wdx))
      values_at_top_right = img(idn,y1,x2,idc);

    if ((y2 >= 0) && (y2 < wdy) && (x1 >= 0) && (x1 < wdx))
      values_at_bot_left  = img(idn,y2,x1,idc);

    if ((y2 >= 0) && (y2 < wdy) && (x2 >= 0) && (x2 < wdx))
      values_at_bot_right = img(idn,y2,x2,idc);

    T horizontal_interpolated_top = values_at_top_left  + (x - x1 ) * (values_at_top_right-values_at_top_left);
    T horizontal_interpolated_bot = values_at_bot_left  + (x - x1 ) * (values_at_bot_right-values_at_bot_left);
    // T interpolated_result = horizontal_interpolated_top + (y - y1 ) * ( horizontal_interpolated_bot - horizontal_interpolated_top);

    if ((y1 >= 0) && (y2 < wdy) && (x1 >= 0) && (x2 < wdx)){

    //   out(idn,idy,idx,idc) = interpolated_result;
    // else
    //   out(idn,idy,idx,idc) = 0;

      // Gradients wrt. the pixel values
      T gIn = gradIn(idn,idy,idx,idc);
      if ((y1 >= 0) && (y1 < wdy) && (x1 >= 0) && (x1 < wdx))
        // gradImg(idn,y1,x1,idc) += gradIn(idn,idy,idx,idc) * (y2 - y) * (x2-x);
        //tensorflow::CudaAtomicAdd
         //tensorflow::CudaAtomicAdd(  &gradImg(idn,y1,x1,idc)  ,  (T) (gIn * (y2 - y) * (x2-x) ) );
         atomicAdd( &gradImg(idn,y1,x1,idc)  ,  (T) (gIn * (y2 - y) * (x2-x) ));

      if ((y1 >= 0) && (y1 < wdy) && (x2 >= 0) && (x2 < wdx))
        // gradImg(idn,y1,x2,idc) += gradIn(idn,idy,idx,idc) * (y2 - y) * (x-x1);
         //tensorflow::CudaAtomicAdd(  &gradImg(idn,y1,x2,idc)  ,  (T) (gIn * (y2 - y) * (x-x1) ) );
         atomicAdd(  &gradImg(idn,y1,x2,idc)  ,  (T) (gIn * (y2 - y) * (x-x1) ) );

      if ((y2 >= 0) && (y2 < wdy) && (x1 >= 0) && (x1 < wdx))
        // gradImg(idn,y2,x1,idc) += gradIn(idn,idy,idx,idc) * (y - y1) * (x2-x);
         //tensorflow::CudaAtomicAdd(  &gradImg(idn,y2,x1,idc)  ,  (T) (gIn * (y - y1) * (x2-x) ) );
         atomicAdd(  &gradImg(idn,y2,x1,idc)  ,  (T) (gIn * (y - y1) * (x2-x) ) );

      if ((y2 >= 0) && (y2 < wdy) && (x2 >= 0) && (x2 < wdx))
        // gradImg(idn,y2,x2,idc) += gradIn(idn,idy,idx,idc) * (y - y1) * (x-x1);
         //tensorflow::CudaAtomicAdd(  &gradImg(idn,y2,x2,idc)  ,  (T) (gIn * (y - y1) * (x-x1) ) );
         atomicAdd(  &gradImg(idn,y2,x2,idc)  ,  (T) (gIn * (y - y1) * (x-x1) ) );

      // Gradients wrt. the coordinates  (add because of colour channels)       
      //tensorflow::CudaAtomicAdd( &gradCoords(idn,1, idy, idx) , (T) (gIn * ( (y2- y) * (values_at_top_right-values_at_top_left)  + (y-y1) *  (values_at_bot_right-values_at_bot_left) ) ) );
      //tensorflow::CudaAtomicAdd( &gradCoords(idn,0, idy, idx) , (T) (gIn * (horizontal_interpolated_bot - horizontal_interpolated_top) ) );
      atomicAdd( &gradCoords(idn,1, idy, idx) , (T) (gIn * ( (y2- y) * (values_at_top_right-values_at_top_left)  + (y-y1) *  (values_at_bot_right-values_at_bot_left) ) ) );
      atomicAdd( &gradCoords(idn,0, idy, idx) , (T) (gIn * (horizontal_interpolated_bot - horizontal_interpolated_top) ) );
    }
  }
};



// Implementing the GPU version of the functor here
template <typename T>
struct MapcoordinatesGradientsFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor  &coords ,
                  const typename Tensor4<T>::ConstTensor  &gradIn ,
                  typename Tensor4<T>::Tensor &gradImg,
                  typename Tensor4<T>::Tensor &gradCoords,
                  int interp_type)
                    {
    const GPUDevice d = context->eigen_device<GPUDevice>();
    const int wdn = img.dimensions()[0];
    const int wdy = img.dimensions()[1];
    const int wdx = img.dimensions()[2];
    const int wdc = img.dimensions()[3];    

    // printf("   IMG (NHWC): %d, %d, %d, %d\n", wdn,wdy ,wdx,wdc);
    // printf("coords (N2HW): %ld, %ld, %ld, %ld\n", coords.dimensions()[0],coords.dimensions()[1] ,coords.dimensions()[2], coords.dimensions()[3]);
    // printf("   out (NHWC): %ld, %ld, %ld, %ld\n",    out.dimensions()[0],   out.dimensions()[1] ,   out.dimensions()[2],    out.dimensions()[3]);


    // unsigned int N = wdn*wdy*wdx*wdc;
    // // initialize gradients with 0s, watch out to tell thrust that the pointer is on the GPU device!
    thrust::fill(thrust::device_ptr<T>(gradImg.data()), 
                 thrust::device_ptr<T>(gradImg.data()+gradImg.size()),
                  T(0));
    thrust::fill(thrust::device_ptr<T>(gradCoords.data()), 
                 thrust::device_ptr<T>(gradCoords.data()+gradCoords.size()),
                  T(0));    

    dim3 block_size(32,32,1);
    int gdx = (std::ceil( wdx /static_cast<float>(block_size.x )));
    int gdy = (std::ceil( wdy /static_cast<float>(block_size.y ))); 
    int gdz = (std::ceil( wdn*wdc /static_cast<float>(block_size.z ))); 
    dim3 grid_size( gdx,gdy,gdz); 

    // printf("   gdx: %d, gdy: %d, gdz:%d \n", gdx,gdy ,gdz);
    // MapcoordinatesGPU<T><<<grid_size,block_size>>>(img ,coords, out,  interp_type);
    if (interp_type == tficg::INTERP_TYPE_BILINEAR){
      MapcoordinatesGradientsGPU<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords,gradIn, gradImg, gradCoords, interp_type);
    }
    else if ((interp_type == tficg::INTERP_TYPE_BICUBIC_2POINTS) or (interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS)){
      // printf(" ERROR not implemented");
      // OP_REQUIRES(context, true == false,
      // tensorflow::errors::Unimplemented("Gradient currently only implemented for Bilinear interpolation on GPU"));
      MapcoordinatesGradientsGPUBicubic<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords,gradIn, gradImg, gradCoords, interp_type);
    }
    else
      printf("\n !!!ERROR UNKNOWN FILTER INTERPOLATION TYPE !!!\n ");

    
    // cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // printf(" kernel calll finished\n");
    // std::cout << cudaerr << std::endl;
    // MapcoordinatesGPU<T><<<grid_size,block_size,0,d.stream()>>>(img ,coords, out,  interp_type);
  }
};


// template struct MapcoordinatesFunctor<GPUDevice, float>
// template struct MapcoordinatesFunctor<GPUDevice, int>

#define REGISTER_GPU_FUNCTOR(T) \
template struct MapcoordinatesGradientsFunctor<GPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU_FUNCTOR);
// TF_CALL_float(REGISTER_GPU_FUNCTOR);
// TF_CALL_double(REGISTER_GPU_FUNCTOR);
// TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_FUNCTOR);
#undef REGISTER_GPU_FUNCTOR





#endif
