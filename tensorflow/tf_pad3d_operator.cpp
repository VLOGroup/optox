///@file tf_pad3d_operator.h
///@brief Tensorflow wrappers for nabla operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 06.2020

#include <vector>

#include "tf_utils.h"
#include "operators/pad3d_operator.h"

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
  int padX, padY, padZ;
  c->GetAttr("pad_x", &padX);
  c->GetAttr("pad_y", &padY);
  c->GetAttr("pad_z", &padZ);

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

  auto in_dim_z = c->Dim( input, 1);
  auto in_dim_y = c->Dim( input, 2);
  auto in_dim_x = c->Dim( input, 3);

  // if the value is known => do a check at graph build time, else at runtime
  if ( c->ValueKnown(in_dim_x) && c->ValueKnown(in_dim_y) && c->ValueKnown(in_dim_z))   {
    //std::cout << "in_dim_y= " << c->Value(in_dim_x)  << ",in_dim_x= " << c->Value(in_dim_x) << ", pad= " << pad;
    if (Transpose){
      if ( !( ( c->Value(in_dim_x)  >= 3*padX) && ( c->Value(in_dim_y) >= 3*padY) && ( c->Value(in_dim_z)  >= 3*padZ) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("PaddingTranspose: The image needs to be bigger than 3x pad (pad+img+pad)! But pad is ",
                                   padX,",",padY,",",padZ, " and x,y,z =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y),",",c->ValueKnown(in_dim_z))  );
      }
    }
    else{
      if ( !( ( c->Value(in_dim_x)  >= padX) && ( c->Value(in_dim_y) >= padY) && ( c->Value(in_dim_z) >= padZ) ) ) {
        // for a given padding size a minimum image size is required => throw error if not satisfied
        TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Padding: The Image needs to be bigger than padding! But pad is ",
                                   padX,",",padY,",",padZ, " and x,y,z =",c->ValueKnown(in_dim_x), ",",c->ValueKnown(in_dim_y), ",",c->ValueKnown(in_dim_z))  );
      }
    }
  }


  if (Transpose){
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 1), 2*padZ , &dims[1]) );
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 2), 2*padY , &dims[2]) );
    TF_RETURN_IF_ERROR( c->Subtract( c->Dim( input, 3), 2*padX , &dims[3]) );
  }
  else{
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 1), 2*padZ , &dims[1]) );    
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 2), 2*padY , &dims[2]) );    
    TF_RETURN_IF_ERROR( c->Add( c->Dim( input, 3), 2*padX , &dims[3]) );
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
REGISTER_OP("Pad3d")
		.Input("x: T")
		.Output("padded_x: T")
		.Attr("T: {float32, float64}")
        .Attr("mode: {'symmetric','reflect','replicate'}")
        .Attr("pad_x: int >= 0")
        .Attr("pad_y: int >= 0")
        .Attr("pad_z: int >= 0")
		.SetShapeFn(PadShapeFn);

REGISTER_OP("Pad3dTranspose")
		.Input("padded_x: T")
		.Output("x: T")
		.Attr("T: {float32, float64}")
        .Attr("mode: {'symmetric','reflect','replicate'}")
        .Attr("pad_x: int >= 0")
        .Attr("pad_y: int >= 0")
        .Attr("pad_z: int >= 0")
		.SetShapeFn(PadTransposeShapeFn);

template <typename T>
class TFPad3dOperator : public OpKernel {
public:
	
	explicit TFPad3dOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
        // Get attributes
        OP_REQUIRES_OK(context, context->GetAttr("pad_x", &padX_));
        OP_REQUIRES_OK(context, context->GetAttr("pad_y", &padY_));
        OP_REQUIRES_OK(context, context->GetAttr("pad_z", &padZ_));
        OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    }

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape output_shape = x_tensor.shape();

        output_shape.set_dim(1, output_shape.dim_size(1) + 2*padZ_);
        output_shape.set_dim(2, output_shape.dim_size(2) + 2*padY_);
        output_shape.set_dim(3, output_shape.dim_size(3) + 2*padX_);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 4>(x_tensor);
		auto output = getDTensorTensorflow<T, 4>(*output_tensor);
		
		optox::Pad3dOperator<T> op(padX_, padX_, padY_, padY_, padZ_, padZ_, mode_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}

     private:
        int padX_;
        int padY_;
        int padZ_;
        std::string mode_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Pad3d") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFPad3dOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);


#undef REGISTER_GPU


template <typename T>
class TFPad3dTransposeOperator : public OpKernel {
public:
	
	explicit TFPad3dTransposeOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
        // Get attributes
        OP_REQUIRES_OK(context, context->GetAttr("pad_x", &padX_));
        OP_REQUIRES_OK(context, context->GetAttr("pad_y", &padY_));
        OP_REQUIRES_OK(context, context->GetAttr("pad_z", &padZ_));
        OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_));
    }

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape output_shape = x_tensor.shape();

        output_shape.set_dim(1, output_shape.dim_size(1) - 2*padZ_);
        output_shape.set_dim(2, output_shape.dim_size(2) - 2*padY_);
        output_shape.set_dim(3, output_shape.dim_size(3) - 2*padX_);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<T, 4>(x_tensor);
		auto output = getDTensorTensorflow<T, 4>(*output_tensor);
		
		optox::Pad3dOperator<T> op(padX_, padX_, padY_, padY_, padZ_, padZ_, mode_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}

     private:
        int padX_;
        int padY_;
        int padZ_;
        std::string mode_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Pad3dTranspose") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFPad3dTransposeOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU