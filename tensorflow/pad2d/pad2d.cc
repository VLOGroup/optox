#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "pad2d.h"
#include "definitions.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

//namespace {



Status PaddingShapeFn(InferenceContext* c, bool Transpose) {
/* Source of the InferenceContext, etc.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h
  More complete usage examples:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/array_ops.cc
*/
  // Get the padding size => handed over as attribute
  int pad;
  c->GetAttr("pad", &pad);


  // Get the padding size => handed over as first input
  ShapeHandle input = c->input(0);

/*  printf ("pad %i  \n", pad);
  fflush(stdout);*/

  // create Dimension Handle to store output dimensions
  std::vector<DimensionHandle> dims(4);
  // Safety Check => Input dimensionality must be of rank 4
  TF_RETURN_IF_ERROR( c->WithRank(input, 4, &input));

  // c->input(idx)  => returns the ShapeHandle for the specified input
  // c->Dim (ShapeHandle, idx)  => returns the size of the dimension as  DimensionHandle 
  //     =>  c->Dim( c->input(0) , 2 )  => will return the 2nd Dimension from the first input
  // c->Add (DimensionHandle first, DimensionOrConstant second, DimensionHandle* out)  => returns a Status
  TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 0), 0     , &dims[0]) );
  TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 1), 0     , &dims[1]) );

  auto in_dim_y = c->Dim( input, 2);
  auto in_dim_x = c->Dim( input, 3);
  // if the value is known => do a check at graph build time, else at runtime
  if ( c->ValueKnown(in_dim_x) && c->ValueKnown(in_dim_y))   {
    //std::cout << "in_dim_y= " << c->Value(in_dim_x)  << ",in_dim_x= " << c->Value(in_dim_x) << ", pad= " << pad;
    if (Transpose){
      if ( !( ( c->Value(in_dim_x)  >= 3*pad) && ( c->Value(in_dim_y) >= 3*pad) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("PaddingTranspose: The image needs to be bigger than 3x pad (pad+img+pad)! But pad is ",
                                   pad, " and x,y =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y))  );
      }
    }
    else{
      if ( !( ( c->Value(in_dim_x)  >= pad) && ( c->Value(in_dim_y) >= pad) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Padding: The Image needs to be bigger than padding! But pad is ",
                                   pad, " and x,y =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y))  );
      }
    }
  }


  if (Transpose){
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 2), 2*pad , &dims[2]) );
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 3), 2*pad , &dims[3]) );
  }
  else{
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 2), 2*pad , &dims[2]) );    
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 3), 2*pad , &dims[3]) );
  }


  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

Status PaddShapeFn(InferenceContext* c) {
  return PaddingShapeFn (c , false);
}

Status PaddTransposeShapeFn(InferenceContext* c) {
  return PaddingShapeFn (c , true);
}


REGISTER_OP("Pad2d")
    .Attr("T: realnumbertype")
    .Input("input: T")
    .Output("output: T")
    .Attr("mode: {'REPLICATE','SYMMETRIC'}")
    .Attr("pad: int >= 0")
    .SetShapeFn(PaddShapeFn)
    .Doc(R"doc(
Pad input wrt. to given type. Uses NCHW dataformat.
output: A padded Tensor.
  output = pad2d(input, type)
)doc");

REGISTER_OP("Pad2dTranspose")
    .Attr("T: realnumbertype")
    .Input("input: T")
    .Output("output: T")
    .Attr("mode: {'REPLICATE','SYMMETRIC'}")
    .Attr("pad: int >= 0")
    .SetShapeFn(PaddTransposeShapeFn)
    .Doc(R"doc(
Transpose padding input wrt. to given type.  Uses NCHW dataformat.
output: A transpose padded (cropped wrt. to correct borderhandling) Tensor.
  output = pad2d_transpose(input, type)
)doc");

template<typename T>
void Pad2dKernelLauncher(const Tensor * in, Tensor * out,
                         const int pad, const tficg::borderMode_t mode);
template<typename T>
void Pad2dTransposeKernelLauncher(const Tensor * in, Tensor * out,
                                  const int pad, const tficg::borderMode_t mode);

template<typename T>
class Pad2dOp : public OpKernel {
 public:
  explicit Pad2dOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get attributes
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad_));
    std::string mode_str;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_str));
    mode_ = tficg::strToBorderMode(mode_str);
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Check dimensionality
    OP_REQUIRES(context, input_tensor.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                        input_tensor.dims(), "d."));

    // Prepare output shape
    auto output_shape = input_tensor.shape();
    auto dims = input_tensor.dims();

    // verify that the image is >= than the amount of pixels we want to pad
    OP_REQUIRES(context, 
                (input_tensor.dim_size(dims-2) >= pad_) && (input_tensor.dim_size(dims-1) >= pad_),
                errors::InvalidArgument("Image needs to be bigger than padding! But pad is ",
                                        pad_, " and x,y =",input_tensor.dim_size(dims-1), ",",input_tensor.dim_size(dims-2))
               );

    output_shape.set_dim(dims-2, output_shape.dim_size(dims-2) + 2*pad_);
    output_shape.set_dim(dims-1, output_shape.dim_size(dims-1) + 2*pad_);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Call the cuda kernel launcher
    Pad2dKernelLauncher<T>(&input_tensor, output_tensor, pad_, mode_);
  }

 private:
  int pad_;
  tficg::borderMode_t mode_;
};

template<typename T>
class Pad2dTransposeOp : public OpKernel {
 public:
  explicit Pad2dTransposeOp(OpKernelConstruction* context) : OpKernel(context)
  {
    // Get attributes
    OP_REQUIRES_OK(context, context->GetAttr("pad", &pad_));
    std::string mode_str;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_str));
    mode_ = tficg::strToBorderMode(mode_str);
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Check dimensionality
    OP_REQUIRES(context, input_tensor.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                        input_tensor.dims(), "d."));

    // Prepare output shape
    auto output_shape = input_tensor.shape();
    auto dims = input_tensor.dims();
    output_shape.set_dim(dims-2, output_shape.dim_size(dims-2) - 2*pad_);
    output_shape.set_dim(dims-1, output_shape.dim_size(dims-1) - 2*pad_);

    // verify that the amount of pixels we want to pad is actually possible
    // TODO: it appears the the current paddingTranspose only works up to a padding
    OP_REQUIRES(context, 
                (input_tensor.dim_size(dims-2) >= 3*pad_) && (input_tensor.dim_size(dims-1) >= 3*pad_),
                errors::InvalidArgument("PaddingTranspose: The image needs to be bigger than 3x pad (pad+img+pad)! But pad is ",
                                        pad_, " and x,y =",input_tensor.dim_size(dims-1), ",",input_tensor.dim_size(dims-2))
               );    

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));

    // Call the cuda kernel launcher
    Pad2dTransposeKernelLauncher<T>(&input_tensor, output_tensor, pad_, mode_);
  }

 private:
  int pad_;
  tficg::borderMode_t mode_;
};

#define REGISTER_GPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("Pad2d") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Pad2dOp<T>) \
REGISTER_KERNEL_BUILDER(Name("Pad2dTranspose") \
                        .Device(DEVICE_GPU) \
                        .TypeConstraint<T>("T"), \
                        Pad2dTransposeOp<T>)

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU_KERNEL);

#undef REGISTER_GPU_KERNEL
