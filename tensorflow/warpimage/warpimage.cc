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
#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "warpimage.h"

// for setting up Shape functions
#include "tensorflow/core/framework/common_shape_fns.h"


// for CPU implementation
#include <vector>       // std::vector
#include <algorithm>




using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Warpimage")
    .Attr("T: realnumbertype")
    .Input("img: T")
    .Input("flow: T")
    .Attr("interp_type: {'BILINEAR','BICUBIC_2POINTS','BICUBIC_4POINTS'} = 'BILINEAR'")
    .Output("output: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
     })    //  #include "tensorflow/core/framework/common_shape_fns.h" 
    .Doc(R"doc(
  Calculates a pixelwise mapped/warped image from the input image, just like scipy.ndimage.filters.map_coordinates.
  The current state of implementation requires that the 3 input variables share the same height, width and batch size.

  Gradient Calculation for BILINEAR is currently implemented wrt. flow.
  For BICUBIC_2POINTS  and BICUBIC_4POINTS ist is implemented against the img part only.

  Args:

      img: a 4D Tensor containing a batch of Images NHWC

      flow:   a 4D Tensor containing a the relative displacement to sample the image from the original image N2HW
              where the second dimension is of size 2 (0 for x and 1 for y movements of the pixel)
              Each batch is interpreted to have its own mapping

      interp_type: ["BILINEAR"] a string that specifies the interpolation method to use
              "BILINEAR", uses bilinear interpolation with 0-padding.
                    Yields identical numerical results to scipy.ndimage.filters.map_coordinates(I,C,mode="constant",order =1)
              "BICUBIC_2POINTS" uses bicubic interpolation with 0 padding and forward & backward differences,  therefore needing 2 points in total.
                    Yields identical numerical results to scipy.ndimage.filters.map_coordinates(I,C,mode="constant",order =2)
              "BICUBIC_4POINTS" uses bicubic interpolation with 0 padding and middle differences at the two left & right point, therefore needing 4 points in total.
                    Yields identical numerical results to scipy.ndimage.filters.map_coordinates(I,C,mode="constant",order =2)
  Returns:
      output: A 4D Tensor containing the filtered image NHWC.

  Examples:
      Simple examples:
        output = warpimage(image,flow)
        output = warpimage(image,flow,"BILINEAR")
        output = warpimage(image,flow,"BICUBIC_4POINTS")


)doc");




template <typename Device, typename T>
class WarpimageOp : public OpKernel {
 public:
  explicit WarpimageOp(OpKernelConstruction* context) : OpKernel(context) {
    // Attributes are only  available in KernelConstruction Phase
    //OP_REQUIRES_OK(context, context->GetAttr("mydebug", &mydebug));
    
    std::string interp_type_str;
    OP_REQUIRES_OK(context, context->GetAttr("interp_type", &interp_type_str));
    interp_type = tficg::strTointerp_type(interp_type_str);
    OP_REQUIRES(context, interp_type != tficg::INTERP_TYPE_INVALID,
        errors::Unimplemented("Not supported interp_type type!"));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor from the context: 
    const Tensor& img_tensor = context->input(0);

    // Convert the Tensor to the Eigen Tensor format :
    //   (Actually a TensorMap, which is a view that behaves like a tensor but doesn't own the memory)
    //   This allows to use the high level Tensor functions -> see the docu here:
    //    (https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md)
    //   examples:   img.setZero(); img.dimensions()[0]; img + img.constant(1.5f); img.setConstant(1); ...
    auto img = img_tensor.tensor<T,4>();
    // Alternatively one can also convert the tensor to a single array and use C syntax for accessing the elements
    // auto input = img_tensor.flat<T>();


    const Tensor& flow_tensor = context->input(1);
    auto flow = flow_tensor.tensor<T,4>();
    // auto flow = flow_tensor.flat<T>();

    // Check the dimensionality and size of the filters and angles
    OP_REQUIRES(context, img_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC) for the images, got ",
                                        img_tensor.dims(), "d."));
    OP_REQUIRES(context, flow_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (N2HW) for the coordinates, got ",
                                        flow_tensor.dims(), "d."));    

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, img_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->tensor<T,4>();
    // auto output = output_tensor->template flat<T>();

    WarpimageFunctor<Device,T>()(context, img, flow, output, interp_type);
  }
    private:
      tficg::interp_type_t interp_type;
};

template <typename T>
void WarpimageCPU(const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor &flow,
                  typename Tensor4<T>::Tensor &out, int interp_type){  
  const int wdn = img.dimensions()[0];
  const int wdy = img.dimensions()[1];
  const int wdx = img.dimensions()[2];
  const int wdc = img.dimensions()[3];
  // std::cout << "   IMG (NHWC):" << wdn <<","<< wdy <<","<< wdx <<","<< wdc <<"\n";
  // std::cout << "flow (N2HW):" << flow.dimensions()[0] <<","<< flow.dimensions()[1] <<","<< flow.dimensions()[2] <<","<< flow.dimensions()[3] <<"\n";
  // std::cout << "   out (NHWC):" << out.dimensions()[0] <<","<< out.dimensions()[1] <<","<< out.dimensions()[2] <<","<< out.dimensions()[3] <<"\n";
  for (int idn=0;idn < wdn; idn++ ){
    for (int idc=0; idc < wdc; idc++){
      for (int idy = 0; idy< wdy; idy++){
        for (int idx = 0; idx< wdx; idx++){
          const T u = flow(idn,0, idy, idx);
          const T v = flow(idn,1, idy, idx);
          const T du = u - std::floor(u);
          const T dv = v - std::floor(v);

          const int x1 = idx +  std::floor(u);
          const int y1 = idy +  std::floor(v);
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

          T horizontal_interpolated_top = values_at_top_left  + du * (values_at_top_right - values_at_top_left);
          T horizontal_interpolated_bot = values_at_bot_left  + du * (values_at_bot_right - values_at_bot_left);
          T interpolated_result = horizontal_interpolated_top + dv * ( horizontal_interpolated_bot - horizontal_interpolated_top);

          if ((y1 >= 0) && (y2 < wdy) && (x1 >= 0) && (x2 < wdx))
            out(idn,idy,idx,idc) = interpolated_result;
          else
            out(idn,idy,idx,idc) = 0;
        }
      }
    }
  }
}

// Implementing the CPU version of the functor here
template <typename T>
struct WarpimageFunctor<CPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor &flow,
                  typename Tensor4<T>::Tensor &out, tficg::interp_type_t interp_type)
                    {
        //WarpimageCPU<T>(img ,flow, out,  interp_type);
    if (interp_type == tficg::INTERP_TYPE_BILINEAR){
       WarpimageCPU<T>(img ,flow, out,  interp_type);
    }
    else if ((interp_type == tficg::INTERP_TYPE_BICUBIC_2POINTS) or (interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS)){
      printf(" CPU ERROR not implemented");
      OP_REQUIRES(context, true == false,
      tensorflow::errors::Unimplemented("Currently only implemented for Bilinear interpolation on CPU"));
    }
    else
      printf("\n !!!ERROR UNKNOWN FILTER INTERPOLATION TYPE !!!\n ");
    


  }
};

// // instantation via functor an function for registration of template => building the template for multiple datatypes
#define REGISTER_CPU_FUNCTOR(T) \
template struct WarpimageFunctor<CPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_CPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(REGISTER_CPU_FUNCTOR);
#undef REGISTER_CPU_FUNCTOR

// register the generated MedianFilter functions to Tensorflow for all registerd datatypes
#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(Name("Warpimage").Device(DEVICE_CPU).TypeConstraint<T>("T"), WarpimageOp<CPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_CPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_CPU);
#undef REGISTER_CPU


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// REGISTER A GPU VERSION, implementation in .cu.cc file
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("Warpimage").Device(DEVICE_GPU).TypeConstraint<T>("T"), WarpimageOp<GPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU);
#undef REGISTER_GPU


// //Explicit registration of datatypes
// REGISTER_KERNEL_BUILDER(Name("Warpimage").Device(DEVICE_GPU).TypeConstraint<float>("T"), WarpimageOp<GPUDevice,float>); 
// REGISTER_KERNEL_BUILDER(Name("Warpimage").Device(DEVICE_GPU).TypeConstraint<double>("T"), WarpimageOp<GPUDevice,double>); 
// REGISTER_KERNEL_BUILDER(Name("Warpimage").Device(DEVICE_GPU).TypeConstraint<int32>("T"), WarpimageOp<GPUDevice,int32>); 






// ###################################################################################################################################################
// GRADIENT SECTION
// ###################################################################################################################################################


REGISTER_OP("WarpimageGradients")
    .Attr("T: realnumbertype")
    .Input("img: T")
    .Input("flow: T")    
    .Input("grad_in: T")
    .Attr("interp_type: {'BILINEAR','BICUBIC_2POINTS','BICUBIC_4POINTS'} = 'BILINEAR'")
    .Output("grad_img: T")
    .Output("grad_flow: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
     })    //  #include "tensorflow/core/framework/common_shape_fns.h" 
    .Doc(R"doc(
Calculates the gradients for the warpimage function.
The current state of implementation requires that the 3 input variables share the same height, width and batch size

  img: a 4D Tensor conatining a batch of Images NHWC

  flow: a 4D Tensor containing the new position to sample the image from the original image N2HW
          where the second dimension is of size 2 (1 for x and 0 for y movements of the pixel)
          Each batch is interpreted to have its own mapping  

  gradIn: a 4D Tensor containing the input gradients comming from the backward path

  gradImg: an output Tensor containing the gradients with respect to the image

  gradFlow: an output Tensor containing the gradients with respect to the coordinates


usage
  (grad_img,grad_flow) = warpimage_gradients(image,flow,grad_in)
)doc");




template <typename Device, typename T>
class WarpimageGradientsOp : public OpKernel {
 public:
  explicit WarpimageGradientsOp(OpKernelConstruction* context) : OpKernel(context) {
    // Attributes are only  available in KernelConstruction Phase
    //OP_REQUIRES_OK(context, context->GetAttr("mydebug", &mydebug));
        std::string interp_type_str;
        OP_REQUIRES_OK(context, context->GetAttr("interp_type", &interp_type_str));
        interp_type = tficg::strTointerp_type(interp_type_str);
        OP_REQUIRES(context, interp_type != tficg::INTERP_TYPE_INVALID,
        errors::Unimplemented("Not supported interp_type type!"));    
      }

  void Compute(OpKernelContext* context) override {
    
    // Grab the input tensor from the context (const tensorflow::Tensor): 
    const Tensor& img_tensor = context->input(0);

    // First check the dimensionality of the input tensro
    OP_REQUIRES(context, img_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC), got ",
                                        img_tensor.dims(), "d."));

    // Then convert the Tensor to the Eigen Tensor format for further processing:
    //   (Actually a TensorMap, which is a view that behaves like a tensor but doesn't own the memory)
    //   This allows to use the high level Tensor functions -> see the docu here:
    //    (https://bitbucket.org/eigen/eigen/src/default/unsupported/Eigen/CXX11/src/Tensor/README.md)
    //   examples:   img.setZero(); img.dimensions()[0]; img + img.constant(1.5f); img.setConstant(1); ...
    auto img = img_tensor.tensor<T,4>();   
    // Alternatively one can also convert the tensor to a single array and use C syntax for accessing the elements
    // auto input = img_tensor.flat<T>();

    const Tensor& flow_tensor = context->input(1);
    OP_REQUIRES(context, flow_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (N2HW) for the coordinates, got ",
                                        flow_tensor.dims(), "d."));    
    auto flow = flow_tensor.tensor<T,4>();


    const Tensor& gradIn_tensor = context->input(2);
    OP_REQUIRES(context, flow_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor (NHWC) for the images, got ",
                                        gradIn_tensor.dims(), "d."));        
    auto gradIn = gradIn_tensor.tensor<T,4>();

    // Create an output tensor
    Tensor* gradImg_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, img_tensor.shape(),
                                                     &gradImg_tensor));
    // Create an output tensor
    Tensor* gradFlow_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, flow_tensor.shape(),
                                                     &gradFlow_tensor));


    auto gradImg = gradImg_tensor->tensor<T,4>();
    auto gradFlow = gradFlow_tensor->tensor<T,4>();
    // auto output = output_tensor->template flat<T>();

    WarpimageGradientsFunctor<Device,T>()(context, img, flow, gradIn, gradImg, gradFlow, interp_type);

  }
    private:
        int interp_type;
};

 //WarpimageGradientsCPU<T>(img, flow, gradIn ,gradImg_tensor, gradFlow_tensor,  interp_type);
template <typename T>
void WarpimageGradientsCPU( 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor  &flow ,
                  const typename Tensor4<T>::ConstTensor  &gradIn ,
                  typename Tensor4<T>::Tensor &gradImg,
                  typename Tensor4<T>::Tensor &gradFlow,
                  int interp_type){  
//
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
//   Gradients wrt. the coordinates (gradFlow)
//   @(x,y)  : dL/dx   = dl/df(R(x,y)) * ( (y2-y)*(f(Q21)-f(Q11)) + (y-y1)*(f(Q22)-f(Q12)) )
//   @(x,y)  : dL/dy   = dl/df(R(x,y)) * ( f(R2)-f(R1) )
//
//   gradIn(x,y) := dl/df(R(x,y)) 

  int wdn = gradIn.dimensions()[0];
  int wdy = gradIn.dimensions()[1];
  int wdx = gradIn.dimensions()[2];
  int wdc = gradIn.dimensions()[3];
  // std::cout << " Calculating Gradient";
  // std::cout << "   IMG (NHWC):" << wdn <<","<< wdy <<","<< wdx <<","<< wdc <<"\n";
  // std::cout << "flow (N2HW):" << gradImg.dimensions()[0] <<","<< gradImg.dimensions()[1] <<","<< gradImg.dimensions()[2] <<","<< gradImg.dimensions()[3] <<"\n";
  // std::cout << "   out (NHWC):" << gradFlow.dimensions()[0] <<","<< gradFlow.dimensions()[1] <<","<< gradFlow.dimensions()[2] <<","<< gradFlow.dimensions()[3] <<"\n";
  // std::cout << std::endl;
  gradFlow.setZero();
  gradImg.setZero();
  // std::cout << "setZero function"<<std::endl;
  // for (int idn=0;idn < wdn; idn++ ){
  //   for (int idy = 0; idy< wdy; idy++){
  //     for (int idx = 0; idx< wdx; idx++){
  //       for (int idc=0; idc < wdc; idc++){
  //         gradImg(idn,idy,idx,idc) = 0; //  (NHWC)
  //       }
  //       gradFlow(idn,0,idy,idx) = 0; //(N2HW)
  //       gradFlow(idn,1,idy,idx) = 0; //(N2HW)
  //     }
  //   }
  // }

  // // examples of Eigen::Tensor functions
  // std::cout << gradFlow + gradFlow.constant(1.5f)<< std::endl;


  for (int idn=0;idn < wdn; idn++ ){
    for (int idc=0; idc < wdc; idc++){
      for (int idy = 0; idy< wdy; idy++){
        for (int idx = 0; idx< wdx; idx++){

          const T u = flow(idn,0, idy, idx);
          const T v = flow(idn,1, idy, idx);
          const T du = u - std::floor(u);
          const T dv = v - std::floor(v); 

          const int x1 = idx + std::floor(u);
          const int y1 = idy + std::floor(v);
          const int x2 = x1+1; //ceil = floor + 1
          const int y2 = y1+1;


          T values_at_top_left   = 0;
          T values_at_top_right  = 0;
          T values_at_bot_left   = 0;
          T values_at_bot_right  = 0;

          if ((y1 >= 0) && (y1 < wdy) && (x1 >= 0) && (x1 < wdx)){
            values_at_top_left  = img(idn,y1,x1,idc);
          }

          if ((y1 >= 0) && (y1 < wdy) && (x2 >= 0) && (x2 < wdx)){
            values_at_top_right = img(idn,y1,x2,idc);
          }

          if ((y2 >= 0) && (y2 < wdy) && (x1 >= 0) && (x1 < wdx)){
            values_at_bot_left  = img(idn,y2,x1,idc);
          }

          if ((y2 >= 0) && (y2 < wdy) && (x2 >= 0) && (x2 < wdx)){
            values_at_bot_right = img(idn,y2,x2,idc);
          }

          T horizontal_interpolated_top = values_at_top_left  + du * (values_at_top_right-values_at_top_left);
          T horizontal_interpolated_bot = values_at_bot_left  + du * (values_at_bot_right-values_at_bot_left);
          // T interpolated_result = horizontal_interpolated_top + (y - y1 ) * ( horizontal_interpolated_bot - horizontal_interpolated_top);

          if ((y1 >= 0) && (y2 < wdy) && (x1 >= 0) && (x2 < wdx)){

          //   out(idn,idy,idx,idc) = interpolated_result;
          // else
          //   out(idn,idy,idx,idc) = 0;

            // Gradients wrt. the pixel values
            T gIn = gradIn(idn,idy,idx,idc);
            if ((y1 >= 0) && (y1 < wdy) && (x1 >= 0) && (x1 < wdx)){
              gradImg(idn,y1,x1,idc) += gIn * (1.0 - dv) * (1.0 - du);
            }

            if ((y1 >= 0) && (y1 < wdy) && (x2 >= 0) && (x2 < wdx)){
              gradImg(idn,y1,x2,idc) += gIn * (1.0 - dv) * (du    );
            }

            if ((y2 >= 0) && (y2 < wdy) && (x1 >= 0) && (x1 < wdx)){
              gradImg(idn,y2,x1,idc) += gIn * (dv    ) * (1.0-du);
            }

            if ((y2 >= 0) && (y2 < wdy) && (x2 >= 0) && (x2 < wdx)){
              gradImg(idn,y2,x2,idc) += gIn * (dv    ) * (du);
            }

            // Gradients wrt. the coordinates  (add because of colour channels)         
            gradFlow(idn,0, idy, idx) += gIn * ( (1.0-dv) * (values_at_top_right-values_at_top_left)  + dv *  (values_at_bot_right-values_at_bot_left));
            gradFlow(idn,1, idy, idx) += gIn * (horizontal_interpolated_bot - horizontal_interpolated_top);
          
          }
        }
      }
    }
  }
};

// Implementing the CPU version of the functor here
template <typename T>
struct WarpimageGradientsFunctor<CPUDevice, T> {
  void operator()(tensorflow::OpKernelContext* context, 
                  const typename Tensor4<T>::ConstTensor  &img ,
                  const typename Tensor4<T>::ConstTensor  &flow ,
                  const typename Tensor4<T>::ConstTensor  &gradIn ,
                  typename Tensor4<T>::Tensor &gradImg,
                  typename Tensor4<T>::Tensor &gradFlow,
                  int interp_type)
                    {
    // thrust::fill(thrust::host_ptr<T>(gradImg.data()), 
    //              thrust::host_ptr<T>(gradImg.data()+gradImg.size()),
    //               T(0));
    // thrust::fill(thrust::host_ptr<T>(gradFlow.data()), 
    //              thrust::host_ptr<T>(gradFlow.data()+gradFlow.size()),
    //               T(0));    

    //WarpimageCPU<T>(img ,flow, out,  interp_type);
    if (interp_type == tficg::INTERP_TYPE_BILINEAR){
       WarpimageGradientsCPU<T>( img, flow, gradIn ,gradImg, gradFlow,  interp_type);
    }
    else if ((interp_type == tficg::INTERP_TYPE_BICUBIC_2POINTS) or (interp_type == tficg::INTERP_TYPE_BICUBIC_4POINTS)){
      printf(" CPU ERROR not implemented");
      OP_REQUIRES(context, true == false,
      tensorflow::errors::Unimplemented("Gradient currently only implemented for Bilinear interpolation on CPU"));
    }
    else
      printf("\n !!!ERROR UNKNOWN FILTER INTERPOLATION TYPE !!!\n ");
   


  }
};

// // instantation via functor an function for registration of template => building the template for multiple datatypes
#define REGISTER_CPU_FUNCTOR(T) \
template struct WarpimageGradientsFunctor<CPUDevice, T>;
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_CPU_FUNCTOR);
//TF_CALL_float(REGISTER_CPU_FUNCTOR);
TF_CALL_INTEGRAL_TYPES(REGISTER_CPU_FUNCTOR);
#undef REGISTER_CPU_FUNCTOR

// register the generated MedianFilter functions to Tensorflow for all registerd datatypes
#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER(Name("WarpimageGradients").Device(DEVICE_CPU).TypeConstraint<T>("T"), WarpimageGradientsOp<CPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_CPU);
//TF_CALL_float(REGISTER_CPU);
TF_CALL_INTEGRAL_TYPES(REGISTER_CPU);
#undef REGISTER_CPU


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// REGISTER A GPU VERSION, implementation in .cu.cc file
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(Name("WarpimageGradients").Device(DEVICE_GPU).TypeConstraint<T>("T"), WarpimageGradientsOp<GPUDevice,T>); 
// for a full list of type macros see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/register_types.h
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GPU);
//TF_CALL_float(REGISTER_GPU);
// TF_CALL_INTEGRAL_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
