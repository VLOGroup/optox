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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "median_filter_4D.h"

// for setting up Shape functions
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("FilterexamplesMedianFilter4d")
    .Attr("T: realnumbertype")
    .Input("input: T")
    .Attr("filtersize: {'3','5'} = '3'")
    .Attr("filtertype: {'SIMPLE','SHAREDMEMORY'} = 'SHAREDMEMORY'")
    .Attr("debug_indices: bool = false")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnchangedShape) // #include "tensorflow/core/framework/common_shape_fns.h"
    .Doc(R"doc(
Does a median filter over the 4D Input image batch (NHWC). 
The filter is applied with respect to the HW dimensions only.
Each colour channel and each image is treated seperately.

  input: a 2D Tensor containing an image to be filtered

  filtersize: a string ("3" or "5") representing the filter kernel size to be used

  filtertype: specify which type of filter implementation to use.
              SIMPLE => plain C++ or CUDA code, "SHAREDMEMORY" optimized CUDA version if on GPU

  debug_indices: if set to true, returns the indices processed internally, instead of the values


output: A 4D Tensor containing the filtered image batch (NHWC).
  output = median_filter(ImgIn,)

  output = median_filter(ImgIn,filtersize = "3")

  output = median_filter(ImgIn,filtertype= "SIMPLE")

)doc");
          //output shape = input shape




template <typename Device, typename T>
class FilterexamplesMedianFilter4dOp : public OpKernel {
 public:
  explicit FilterexamplesMedianFilter4dOp(OpKernelConstruction* context) : OpKernel(context) {
    // Attributes are only  available in KernelConstruction Phase
    OP_REQUIRES_OK(context, context->GetAttr("debug_indices", &debug_indices));
    
    std::string strFiltersize;
    OP_REQUIRES_OK(context, context->GetAttr("filtersize", &strFiltersize));
    if (strFiltersize == "3")
        filtersize = 3;
    else if (strFiltersize == "5")
        filtersize = 5;
    else
        errors::Unimplemented("Not supported INTERPOLATION type!");

    std::string filtertype_str;
    OP_REQUIRES_OK(context, context->GetAttr("filtertype", &filtertype_str));
    filtertype = medianfilter::strToFiltertype(filtertype_str);
    OP_REQUIRES(context, filtertype != medianfilter::FILTER_TYPE_INVALID,
        errors::Unimplemented("Not supported filtertype type!"));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Check the dimensionality and size of the filters and angles
    OP_REQUIRES(context, input_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC), got ",
                                        input_tensor.dims(), "d."));
    auto input = input_tensor.tensor<T,4>();


    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->tensor<T,4>();

    FilterexamplesMedianFilter4dFunctor<Device,T>()(context, input, output, filtersize, filtertype, debug_indices);
  }
    private:
        bool debug_indices;
        int32 filtersize;
        int filtertype;
};





//Explicit registration of the Op with Tensorflow for the desired datatypes
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4d").Device(DEVICE_GPU).TypeConstraint<int32>("T"), FilterexamplesMedianFilter4dOp<GPUDevice,int32>); 
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4d").Device(DEVICE_GPU).TypeConstraint<float>("T"), FilterexamplesMedianFilter4dOp<GPUDevice,float>); 
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4d").Device(DEVICE_GPU).TypeConstraint<double>("T"), FilterexamplesMedianFilter4dOp<GPUDevice,double>); 


// Register the templatatized Op with tensorflow (This requires, that the GPU Functor that is used inside the OP, is already instantated inside the GPU .cu.cc code!!!)
#define REGISTER_OP_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4d").Device(DEVICE_GPU).TypeConstraint<T>("T"), FilterexamplesMedianFilter4dOp<GPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_OP_GPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_OP_GPU);
#undef REGISTER_OP_GPU









//##################################################################################################################################################################
// NEW section for CPU implementation
//##################################################################################################################################################################


#include <vector>       // std::vector
#include <algorithm>

template <typename T>
void sort( std::vector<T> &mysortbuffer,  int size)
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
void FilterexamplesMedianFilter4dCPU(
  const typename tensorflow::TTypes<T,4>::ConstTensor &img,
  typename tensorflow::TTypes<T,4>::Tensor &out, 
  int filtersize, int filtertype, bool debug_indices){
  
  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  const int RES = (filtersize-1)/2;
  const int MED_BUFFER_HALF = std::floor((filtersize*filtersize)/2);
  for (int idn =0; idn < wdn; idn++){
    for (int idc =0; idc < wdc; idc++){
      for (int idx =0; idx < wdx; idx++){
        for (int idy =0; idy < wdy; idy++){
  
          if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){      
            // this if is actually just necessary in GPU code
            std::vector<T> mysortbuffer (filtersize*filtersize);

            for (int j =0; j<(filtersize*filtersize);j++){
              mysortbuffer.at(j) = 0; // Initialize all with 0 
            }

            for ( int dx = 0; dx < filtersize; dx++){
              for ( int dy = 0; dy < filtersize; dy++){
                int idx2 = idx + dx - RES;
                int idy2 = idy + dy - RES;
                if ((idx2 >= 0) && (idy2 >= 0) && (idx2 < wdx) && (idy2 < wdy) ){
                    mysortbuffer.at(dx + filtersize*dy) = img(idn,idy2,idx2,idc);
                }
              }         
            }
            // sort & select the median
            // std::sort(mysortbuffer.begin(),mysortbuffer.end());
            sort<T>(mysortbuffer,filtersize*filtersize);
            out(idn,idy,idx,idc) = mysortbuffer.at(MED_BUFFER_HALF);

            if (debug_indices){
              out(idn,idy,idx,idc) = idx + idy*wdx ;
            }
          }
        }
      }
    }
  }
};



// Implementing the CPU version of the functor here
template <typename T>
struct FilterexamplesMedianFilter4dFunctor<CPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
    const typename tensorflow::TTypes<T,4>::ConstTensor &img,
    typename tensorflow::TTypes<T,4>::Tensor &out, 
    int filtersize, int filtertype, bool debug_indices) {

    FilterexamplesMedianFilter4dCPU<T>(img,  out, filtersize, filtertype, debug_indices);

  }
};

// Instantiate the templatatized Functor, so that g++ compiles it for the given datatypes:
#define INSTANTIATE_CPU_FUNCTOR(T) \
template struct FilterexamplesMedianFilter4dFunctor<CPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(INSTANTIATE_CPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(INSTANTIATE_CPU_FUNCTOR);
#undef INSTANTIATE_CPU_FUNCTOR

// Register the templatatized Op with tensorflow
#define REGISTER_OP_CPU(T) \
REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4d").Device(DEVICE_CPU).TypeConstraint<T>("T"), FilterexamplesMedianFilter4dOp<CPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_OP_CPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_OP_CPU);
#undef REGISTER_OP_CPU




 // DT_INT8
 // DT_UINT8
 // DT_INT16
 // DT_UINT16
 // DT_INT32
 // DT_INT64
 // DT_DOUBLE
 // DT_FLOAT



//##################################################################################################################################################################
// NEW section for GRADIENTS
//##################################################################################################################################################################
REGISTER_OP("FilterexamplesMedianFilter4dGradient")
    .Attr("T: realnumbertype")
    .Input("input: T")
    .Input("gradin: T")
    .Attr("filtersize: {'3','5'} = '3'")
    .Attr("filtertype: {'SIMPLE','SHAREDMEMORY'} = 'SHAREDMEMORY'")
    .Attr("debug_indices: bool = false")
    .Output("gradout: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
     })    //  #include "tensorflow/core/framework/common_shape_fns.h" 
    .Doc(R"doc(
Calculates the gradient for the backward pass through the median filter.

  input: a 4D Tensor (NHWC) that contains the input image that was filtered in the forward pass
  
  gradin: the gradients recieved at the output from the next Op

  filtersize: a string ("3" or "5") representing the filter kernel size to be used

  filtertype: specify which type of filter implementation to use.
              SIMPLE => plain C++ or CUDA code, "SHAREDMEMORY" optimized CUDA version if on GPU

  debug_indices: if set to true, returns the indices processed internally, instead of the values


  gradout: A 4D Tensor containing the gradients backward propagated throughout the median filter 
  
)doc");



template <typename Device, typename T>
class FilterexamplesMedianFilter4dGradientOp : public OpKernel {
 public:
  explicit FilterexamplesMedianFilter4dGradientOp(OpKernelConstruction* context) : OpKernel(context) {
    // Attributes are only  available in KernelConstruction Phase
    OP_REQUIRES_OK(context, context->GetAttr("debug_indices", &debug_indices));
    
    std::string strFiltersize;
    OP_REQUIRES_OK(context, context->GetAttr("filtersize", &strFiltersize));
    if (strFiltersize == "3")
        filtersize = 3;
    else if (strFiltersize == "5")
        filtersize = 5;
    else
        errors::Unimplemented("Not supported INTERPOLATION type!");

    std::string filtertype_str;
    OP_REQUIRES_OK(context, context->GetAttr("filtertype", &filtertype_str));
    filtertype = medianfilter::strToFiltertype(filtertype_str);
    OP_REQUIRES(context, filtertype != medianfilter::FILTER_TYPE_INVALID,
        errors::Unimplemented("Not supported filtertype type!"));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    const Tensor& gradin_tensor = context->input(1);


    // Check the dimensionality and size of the filters and angles
    OP_REQUIRES(context, input_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC) for the image input, got ",
                                        input_tensor.dims(), "d."));
    auto img = input_tensor.tensor<T,4>();


    // Check the dimensionality and size of the filters and angles
    OP_REQUIRES(context, gradin_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC) for the gradient input, got ",
                                        input_tensor.dims(), "d."));    
    auto gradin = gradin_tensor.tensor<T,4>();

    // Create an output tensor
    Tensor* gradout_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &gradout_tensor));
    auto gradout = gradout_tensor->tensor<T,4>();
    // gradout.setZero();

    FilterexamplesMedianFilter4dGradientFunctor<Device,T>()(context, img, gradin, gradout, filtersize, filtertype, debug_indices);
  }
    private:
        bool debug_indices;
        int32 filtersize;
        int filtertype;
};






// CPU Implementation
template <typename T>
void sort_indices( std::vector<T> &mysortbuffer,  std::vector<int> &idxbuffer, int size)
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





template <typename T>
void FilterexamplesMedianFilter4dGradientCPU(
  const typename tensorflow::TTypes<T,4>::ConstTensor &img,
  const typename tensorflow::TTypes<T,4>::ConstTensor &gradin,
  typename tensorflow::TTypes<T,4>::Tensor &gradout, 
  int filtersize, int filtertype, bool debug_indices) {
  /*
  This function propagates the gradients from the output of the gradient function to its inputs.
  */
  const unsigned int wdn = img.dimensions()[0];
  const unsigned int wdy = img.dimensions()[1];
  const unsigned int wdx = img.dimensions()[2];
  const unsigned int wdc = img.dimensions()[3];

  // for (int i =0;i <N; i++)
  //   gradout[i] = 0;

  const int RES = (filtersize-1)/2;
  const int MED_BUFFER_HALF = std::floor((filtersize*filtersize)/2);
  for (int idn =0; idn < wdn; idn++){
    for (int idc =0; idc < wdc; idc++){
      for (int idx =0; idx < wdx; idx++){
        for (int idy =0; idy < wdy; idy++){
  
          if ( (idn<wdn) && (idc<wdc) && (idy<wdy) && (idx<wdx) ){      
            // this if is actually just necessary in GPU code
            std::vector<T> mysortbuffer (filtersize*filtersize);
            std::vector<int> myidxbuffer (filtersize*filtersize);
            for (int j =0; j<(filtersize*filtersize);j++){
              mysortbuffer.at(j) = 0; // Initialize all with 0 
              myidxbuffer.at(j) = -1; // => all outside 
            }

            for ( int dx = 0; dx < filtersize; dx++){
              for ( int dy = 0; dy < filtersize; dy++){
                int idx2 = idx + dx - RES;
                int idy2 = idy + dy - RES;
                if ( (idx2 >= 0) && (idy2 >= 0) && (idx2 < wdx) && (idy2 < wdy) ){
                    mysortbuffer.at(dx + filtersize*dy) = img(idn,idy2,idx2,idc);
                    myidxbuffer.at(dx + filtersize*dy) = idx2 + idy2*wdx;
                }
              }        
            }
            // sort & select the median
            sort_indices<T>(mysortbuffer,myidxbuffer,filtersize*filtersize);
            // T median =  mysortbuffer.at(MED_BUFFER_HALF);
            const int origId = myidxbuffer.at(MED_BUFFER_HALF);
            const int origx = origId % wdx;
            const int origy = origId / wdx;

            if (origId != -1){
                // gradout needs to be initialized with 0!
                gradout(idn,origy,origx,idc) +=  gradin(idn,idy,idx,idc);

            }
            if (debug_indices){
              gradout(idn,idy,idx,idc) = idx + idy*wdx;
            }
          }
        }
      }
    }
  }
};


template <typename T>
struct FilterexamplesMedianFilter4dGradientFunctor<CPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
  const typename tensorflow::TTypes<T,4>::ConstTensor &img,
  const typename tensorflow::TTypes<T,4>::ConstTensor &gradin,
  typename tensorflow::TTypes<T,4>::Tensor &gradout, 
  int filtersize, int filtertype, bool debug_indices) {
    // thrust::fill(thrust::host_ptr<T>(gradout), thrust::host_ptr<T>(gradout+N), T(0));

    gradout.setZero();

    FilterexamplesMedianFilter4dGradientCPU<T>(img, gradin, gradout, filtersize, filtertype, debug_indices);
    }
};

// Instantiate the templatatized Functor, so that g++ compiles it for the given datatypes:
#define INSTANTIATE_CPU_FUNCTOR(T) \
template struct FilterexamplesMedianFilter4dGradientFunctor<CPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(INSTANTIATE_CPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(INSTANTIATE_CPU_FUNCTOR);
#undef INSTANTIATE_CPU_FUNCTOR

// Register the templatatized Op with tensorflow (This requires, that the GPU Functor that is used inside the OP, is already instantated inside the GPU .cu.cc code!!!)
#define REGISTER_OP_CPU(T) \
REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4dGradient").Device(DEVICE_CPU).TypeConstraint<T>("T"), FilterexamplesMedianFilter4dGradientOp<CPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_OP_CPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_OP_CPU);
#undef REGISTER_OP_CPU
// END CPU Implementation and Registration
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Explicit registration of the Op with Tensorflow for the desired datatypes
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4dGradient").Device(DEVICE_GPU).TypeConstraint<int32>("T"), FilterexamplesMedianFilter4dGradientOp<GPUDevice,int32>); 
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4dGradient").Device(DEVICE_GPU).TypeConstraint<float>("T"), FilterexamplesMedianFilter4dGradientOp<GPUDevice,float>); 
// REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4dGradient").Device(DEVICE_GPU).TypeConstraint<double>("T"), FilterexamplesMedianFilter4dGradientOp<GPUDevice,double>); 


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Register GPU Implementation (implemented in the .cu.cc file)
#define REGISTER_OP_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("FilterexamplesMedianFilter4dGradient").Device(DEVICE_GPU).TypeConstraint<T>("T"), FilterexamplesMedianFilter4dGradientOp<GPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_OP_GPU);
// TF_CALL_INTEGRAL_TYPES(REGISTER_OP_GPU); // only float,double and int32 are available with attomic add on GPU
TF_CALL_int32(REGISTER_OP_GPU);
#undef REGISTER_OP_GPU