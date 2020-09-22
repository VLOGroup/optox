///@file tf_pad2d_operator.h
///@brief Tensorflow wrappers for nabla operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 06.2020

#include <vector>

#include "tf_utils.h"
#include "operators/pad2d_operator.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "tensorflow/core/framework/register_types.h"
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>


using namespace tensorflow;
using namespace std;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

Status TPadShapeFn(InferenceContext* c, bool Transpose) {
/* Source of the InferenceContext, etc.
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h
  More complete usage examples:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/array_ops.cc
*/
  // Get the padding size => handed over as attribute
  int left, right, top, bottom;
  c->GetAttr("left", &left);
  c->GetAttr("right", &right);
  c->GetAttr("top", &top);
  c->GetAttr("bottom", &bottom);

  // Get the padding size => handed over as first input
  ShapeHandle input = c->input(0);

/*  printf ("pad %i  \n", pad);
  fflush(stdout);*/

  // create Dimension Handle to store output dimensions
  std::vector<DimensionHandle> dims(3);
  // Safety Check => Input dimensionality must be of rank 3
  TF_RETURN_IF_ERROR( c->WithRank(input, 3, &input));

  // c->input(idx)  => returns the ShapeHandle for the specified input
  // c->Dim (ShapeHandle, idx)  => returns the size of the dimension as  DimensionHandle 
  //     =>  c->Dim( c->input(0) , 2 )  => will return the 2nd Dimension from the first input
  // c->Add (DimensionHandle first, DimensionOrConstant second, DimensionHandle* out)  => returns a Status
  TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 0), 0     , &dims[0]) );

  auto in_dim_y = c->Dim( input, 1);
  auto in_dim_x = c->Dim( input, 2);

  // if the value is known => do a check at graph build time, else at runtime
  if ( c->ValueKnown(in_dim_x) && c->ValueKnown(in_dim_y))   {
    //std::cout << "in_dim_y= " << c->Value(in_dim_x)  << ",in_dim_x= " << c->Value(in_dim_x) << ", pad= " << pad;
    if (Transpose){
      if ( !( ( c->Value(in_dim_x)  >= 1.5 * (left + right)) && 
              ( c->Value(in_dim_y) >=  1.5 * (top + bottom)) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("PaddingTranspose: The image needs to be bigger than 1.5x (pad0+pad1) (pad0+img+pad1)! But pad is ",
                                   left,",",right,",",
                                   top,",",bottom,
                                   " and x,y =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y) ));
      }
    }
    else{
      if ( !( ( c->Value(in_dim_x)  >= (left + right)) && 
              ( c->Value(in_dim_y) >= (top + bottom)) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Padding: The Image needs to be bigger than padding! But pad is ",
                                   left,",",right,",",
                                   top,",",bottom,
                                   " and x,y =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y) ));
      }
    }
  }


  if (Transpose){
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 1), (top + bottom) , &dims[1]) );
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 2), (left + right) , &dims[2]) );
  }
  else{   
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 1), (top + bottom) , &dims[1]) );    
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 2), (left + right) , &dims[2]) );
  }


  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

Status PadShapeFn(InferenceContext* c) {
  return TPadShapeFn (c , false);
}

Status PadTransposeShapeFn(InferenceContext* c) {
  return TPadShapeFn (c , true);
}

/**
 * register the operation with necessary options
 */
REGISTER_OP("Pad2d")
		.Input("x: T")
		.Output("padded_x: T")
		.Attr("T: {float32, float64}")
        .Attr("mode: {'symmetric','reflect','replicate'}")
        .Attr("left: int >= 0")
        .Attr("right: int >= 0")
        .Attr("top: int >= 0")
        .Attr("bottom: int >= 0")
		.SetShapeFn(PadShapeFn);

REGISTER_OP("Pad2dTranspose")
		.Input("padded_x: T")
		.Output("x: T")
		.Attr("T: {float32, float64}")
        .Attr("mode: {'symmetric','reflect','replicate'}")
        .Attr("left: int >= 0")
        .Attr("right: int >= 0")
        .Attr("top: int >= 0")
        .Attr("bottom: int >= 0")
		.SetShapeFn(PadTransposeShapeFn);

template <typename T>
class TFPad2dOperator : public OpKernel {
public:
	
	explicit TFPad2dOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
        // Get attributes
        OP_REQUIRES_OK(context, context->GetAttr("left", &left_));
        OP_REQUIRES_OK(context, context->GetAttr("right", &right_));
        OP_REQUIRES_OK(context, context->GetAttr("top", &top_));
        OP_REQUIRES_OK(context, context->GetAttr("bottom", &bottom_));
        OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    }

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape output_shape = x_tensor.shape();

        output_shape.set_dim(1, output_shape.dim_size(1) + top_ + bottom_);
        output_shape.set_dim(2, output_shape.dim_size(2) + left_ + right_);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 3>(x_tensor);
		auto output = getDTensorTensorflow<T, 3>(*output_tensor);
		
		optox::Pad2dOperator<T> op(left_, right_, bottom_, top_, mode_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}

     private:
        int left_;
        int right_;
        int bottom_;
        int top_;
        std::string mode_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Pad2d") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFPad2dOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);


#undef REGISTER_GPU


template <typename T>
class TFPad2dTransposeOperator : public OpKernel {
public:
	
	explicit TFPad2dTransposeOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
        // Get attributes
        OP_REQUIRES_OK(context, context->GetAttr("left", &left_));
        OP_REQUIRES_OK(context, context->GetAttr("right", &right_));
        OP_REQUIRES_OK(context, context->GetAttr("top", &top_));
        OP_REQUIRES_OK(context, context->GetAttr("bottom", &bottom_));
        OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    }

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape output_shape = x_tensor.shape();

        output_shape.set_dim(1, output_shape.dim_size(1) - bottom_ - top_);
        output_shape.set_dim(2, output_shape.dim_size(2) - left_ - right_);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 3>(x_tensor);
		auto output = getDTensorTensorflow<T, 3>(*output_tensor);
		
		optox::Pad2dOperator<T> op(left_, right_, bottom_, top_, mode_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}

     private:
        int left_;
        int right_;
        int bottom_;
        int top_;
        std::string mode_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Pad2dTranspose") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFPad2dTransposeOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU