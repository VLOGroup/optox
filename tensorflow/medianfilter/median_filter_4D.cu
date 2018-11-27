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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "median_filter_4D.h"
#include "atomicAddFloat64.h"

#define MAXBUFFERSIZE 5*5
#define MEDSIZE filtersize
#define MED_BUFFER_HALF ( (int) (filtersize*filtersize)/2 )

// tensorflow::CudaAtomicAdd => requires compute capability >= 3.0
// if using nvcc directly specify a compute capability >= 3.0 with the following flags -gencode arch=compute_30,code=sm_30
//#include "tensorflow/core/util/cuda_kernel_helper.h"

// use CUDA Specialized GPU functions that provide STL similar function on GPU
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
__device__ void sort( T * mysortbuffer,int size)
{
  T tmp;
  for (int j =0; j<(size-1);j++){ // -1 because buffer[i+1] <-> buffer[i]
    for (int i =0; i<(size-1-j);i++){
      if (mysortbuffer[i] < mysortbuffer[i+1]){
        tmp = mysortbuffer [i]; 
        mysortbuffer[i] = mysortbuffer[i+1];
        mysortbuffer[i+1] = tmp;
      }
    }
  }
};

template <typename T>
__global__ void FilterexamplesMedianFilter4dSimple(const typename tensorflow::TTypes<T,4>::ConstTensor img,
                                           typename tensorflow::TTypes<T,4>::Tensor out, 
                                         int filtersize, int filtertype, bool debug_indices) {
  // ATTENTION: Tensor object must not be passed via reference into CUDA Kernel!
  // => Tensor DATA is on GPU but extra information like .dimensions() is on CPU heap => invalid pointers
  // => pass by value copies this extra informaiton from CPU to GPU 
  /*
  A simple version of the Median Filter.
  Directly accesses Global Memory => slower than a version that would use shared memory for thread block
  */

  const int RES = (filtersize-1)/2;

  const unsigned int idx = blockIdx.x * (blockDim.x) + threadIdx.x;
  const unsigned int idy = blockIdx.y * (blockDim.y) + threadIdx.y;
  const unsigned int idz = blockIdx.z * (blockDim.z) + threadIdx.z;

  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  // Cuda only has 3 dimensions => split idz into 2 dimensions 
  const unsigned int idn =  idz / wdc;
  const unsigned int idc =  idz % wdc;  



  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){     
    // build a buffer for the median
    T mysortbuffer [MAXBUFFERSIZE];

    for (int j =0; j<(MEDSIZE*MEDSIZE);j++){
      mysortbuffer[j] = 0; // Initialize all with 0 
    } 

    for ( int dx = 0; dx < MEDSIZE; dx++){
      for ( int dy = 0; dy < MEDSIZE; dy++){
        int idx2 = idx + dx - RES;
        int idy2 = idy + dy - RES;
        if ( (idx2 >= 0) && (idy2 >= 0) &&  (idx2 < wdx) && (idy2 < wdy) ){
          mysortbuffer[dx + MEDSIZE*dy] = img(idn,idy2,idx2,idc);
        }
      }
    }
    // sort & select the median
    sort<T>(mysortbuffer,MEDSIZE*MEDSIZE);
    out(idn,idy,idx,idc) = mysortbuffer[MED_BUFFER_HALF];

    if (debug_indices){
      out(idn,idy,idx,idc) = idx + idy * wdx;
    }
  }
};


//###################################################
// Functor Template:
template <typename T>
struct FilterexamplesMedianFilter4dFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
    const typename tensorflow::TTypes<T,4>::ConstTensor &img,
    typename tensorflow::TTypes<T,4>::Tensor &out, 
    int filtersize, int filtertype, bool debug_indices){

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

    cudaEvent_t start, stop;
    
    if (debug_indices){

      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
    }

    FilterexamplesMedianFilter4dSimple<T><<<grid_size,block_size,0,d.stream()>>>(img, out, filtersize, filtersize, debug_indices);
     
    if (debug_indices){
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float time;
      cudaEventElapsedTime(&time, start, stop);
      printf ("Time for the kernel: %f ms\n", time);
    }
  }
};


// // // explicit instantation example with the templatatized Functor, so that g++ compiles it for the given datatypes:
// // template struct FilterexamplesMedianFilter4dFunctor<GPUDevice, float>;
// // template struct FilterexamplesMedianFilter4dFunctor<GPUDevice, doulbe>;
// // template struct FilterexamplesMedianFilter4dFunctor<GPUDevice, tensorflow::int32>;

// Instantiate the templatatized Functor, so that g++ compiles it for the given datatypes:
#define INSTANTIATE_GPU_FUNCTOR(T) \
template struct FilterexamplesMedianFilter4dFunctor<GPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(INSTANTIATE_GPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(INSTANTIATE_GPU_FUNCTOR);
#undef INSTANTIATE_GPU_FUNCTOR


// // // explicit instantation directly with the function, so that g++ compiles it for the given datatypes:
// //void FilterexamplesMedianFilter4dKernelLauncher(const  T* in, const int wdy, const int wdx,  T* out ,int filtersize, int filtertype, bool debug_indices) 
// template void FilterexamplesMedianFilter4dKernelLauncher<float>(const  float* in, const int wdy, const int wdx,  float* out ,int filtersize, int filtertype, bool debug_indices);
// template void FilterexamplesMedianFilter4dKernelLauncher<double>(const  double* in, const int wdy, const int wdx,  double* out ,int filtersize, int filtertype, bool debug_indices);
// template void FilterexamplesMedianFilter4dKernelLauncher<tensorflow::int32>(const  tensorflow::int32* in, const int wdy, const int wdx,  tensorflow::int32* out ,int filtersize, int filtertype, bool debug_indices);



// ###################################################################################################################################################
// GRADIENT SECTION
// ###################################################################################################################################################


template <typename T>
__device__ void sort_indices( T * mysortbuffer,int * idxbuffer,int size)
{
  T tmp;
  int idxtmp;
  for (int j =0; j<(size-1);j++){ // -1 because buffer[i+1] <-> buffer[i]
    for (int i =0; i<(size-1-j);i++){
      if (mysortbuffer[i] < mysortbuffer[i+1]){
        tmp = mysortbuffer [i]; 
        mysortbuffer[i] = mysortbuffer[i+1];
        mysortbuffer[i+1] = tmp;

        idxtmp = idxbuffer[i];
        idxbuffer[i] = idxbuffer[i+1];
        idxbuffer[i+1] =idxtmp;
      }
    }
  }
};


// template<class T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr >
// __device__ void myAttomicAdd(T *pValue, T addValue )
// {
//   // called for floating point types
//   tensorflow::CudaAtomicAdd(pValue,addValue);
// }

// template<class T, typename std::enable_if<std::is_signed<T>::value &&
//                                           !std::is_floating_point<T>::value>::type* = nullptr >
// __device__ void  myAttomicAdd(T *pValue, T addValue )
// {
//   // called for signed integral types
//   atomicAdd(pValue,addValue);
// }


template <typename T>
__global__ void FilterexamplesMedianFilter4dGradient( 
                  const typename tensorflow::TTypes<T,4>::ConstTensor img,
                  const typename tensorflow::TTypes<T,4>::ConstTensor gradin,
                  typename tensorflow::TTypes<T,4>::Tensor gradout, 
                  int filtersize, int filtertype, bool debug_indices) {

  // ATTENTION: Tensor object must not be passed via reference into CUDA Kernel!
  // => Tensor DATA is on GPU but extra information like .dimensions() is on CPU heap => invalid pointers
  // => pass by value copies this extra informaiton from CPU to GPU 
  /*
  This function propagates the gradients from the output of the gradient function to its inputs.
  */
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

  const int RES = (filtersize-1)/2;

  if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){
    // build a buffer for the median
    T mysortbuffer [MAXBUFFERSIZE];
    int myidxbuffer [MAXBUFFERSIZE];

    for (int j =0; j<(MEDSIZE*MEDSIZE);j++){
      mysortbuffer[j] = 0; // Initialize all with 0 
      myidxbuffer[j] = -1; // => all 
    } 

    for ( int dx = 0; dx < MEDSIZE; dx++){
      for ( int dy = 0; dy < MEDSIZE; dy++){
        int idx2 = idx + dx - RES;
        int idy2 = idy + dy - RES;
        if ( (idx2 >= 0) && (idy2 >= 0) &&  (idx2 < wdx) && (idy2 < wdy) ){
          int locId = dx + MEDSIZE*dy;
          mysortbuffer[locId] = img(idn,idy2,idx2,idc);
          myidxbuffer[locId] = idx2 + idy2 * wdx;
        }
      }
    }
    sort_indices<T>(mysortbuffer,myidxbuffer,MEDSIZE*MEDSIZE);
    const int origId = myidxbuffer[MED_BUFFER_HALF] ;
    const int origx = origId  % wdx ;
    const int origy = origId  / wdx ;

    if (origId != -1){
      // requieres that the out variable was initialized with 0
      // gradout[origId] +=  gradin[id];
      atomicAdd ( & gradout(idn,origy,origx,idc) , gradin(idn,idy,idx,idc) );
      // the median filters output  out[id] came from the in[origId]
      // Gradients flow in reverse order
    }

    if (debug_indices){
      gradout(idn,idy,idx,idc) = idx + idy*wdx;
    }
  }
}

//using namespace tensorflow;
template <typename T>
struct FilterexamplesMedianFilter4dGradientFunctor<GPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename tensorflow::TTypes<T,4>::ConstTensor &img,
                  const typename tensorflow::TTypes<T,4>::ConstTensor &gradin,
                  typename tensorflow::TTypes<T,4>::Tensor &gradout, 
                  int filtersize, int filtertype, bool debug_indices) {    
    const GPUDevice d = context->eigen_device<GPUDevice>();
    const int wdn = img.dimensions()[0];
    const int wdy = img.dimensions()[1];
    const int wdx = img.dimensions()[2];
    const int wdc = img.dimensions()[3];    

    // printf("   IMG (NHWC): %d, %d, %d, %d\n", wdn,wdy ,wdx,wdc);
    // printf("coords (N2HW): %ld, %ld, %ld, %ld\n", coords.dimensions()[0],coords.dimensions()[1] ,coords.dimensions()[2], coords.dimensions()[3]);
    // printf("   out (NHWC): %ld, %ld, %ld, %ld\n",    out.dimensions()[0],   out.dimensions()[1] ,   out.dimensions()[2],    out.dimensions()[3]);



    // initialize gradients with 0s, watch out to tell thrust that the pointer is on the GPU device!
    thrust::fill(thrust::device_ptr<T>(gradout.data()),
                 thrust::device_ptr<T>(gradout.data()+gradout.size()), T(0));

    dim3 block_size(32,32,1);
    int gdx = (std::ceil( wdx /static_cast<float>(block_size.x )));
    int gdy = (std::ceil( wdy /static_cast<float>(block_size.y ))); 
    int gdz = (std::ceil( wdn*wdc /static_cast<float>(block_size.z ))); 
    dim3 grid_size( gdx,gdy,gdz); 

    cudaEvent_t start, stop;
    
    if (debug_indices){
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start, 0);
    }

    FilterexamplesMedianFilter4dGradient<T><<<grid_size,block_size,0,d.stream()>>>(img, gradin, gradout, filtersize, filtertype, debug_indices);
     
    if (debug_indices){
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      float time;
      cudaEventElapsedTime(&time, start, stop);
      printf ("Time for the kernel: %f ms\n", time);
    }
  }
};


// // // explicit instantation example with the templatatized Functor, so that g++ compiles it for the given datatypes:
// // template struct FilterexamplesMedianFilter4dGradientFunctor<GPUDevice, float>;
// // template struct FilterexamplesMedianFilter4dGradientFunctor<GPUDevice, doulbe>;
// // template struct FilterexamplesMedianFilter4dGradientFunctor<GPUDevice, tensorflow::int32>;

// Instantiate the templatatized Functor, so that g++ compiles it for the given datatypes:
#define INSTANTIATE_GPU_FUNCTOR(T) \
template struct FilterexamplesMedianFilter4dGradientFunctor<GPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(INSTANTIATE_GPU_FUNCTOR);
// TF_CALL_INTEGRAL_TYPES(INSTANTIATE_GPU_FUNCTOR); // only float,double and int32 are available with attomic add on GPU
TF_CALL_int32(INSTANTIATE_GPU_FUNCTOR);
#undef INSTANTIATE_GPU_FUNCTOR



#endif