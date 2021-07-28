///@file tf_warp_operator.h
///@brief Tensorflow wrappers for warp operator
///@author Kerstin Hammernik <k.hammernik@imperial.ac.uk>
///@date 01.2021

#include <vector>

#include "tf_utils.h"
#include "operators/warp_operator.h"

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

/**
 * register the operation with necessary options
 */
REGISTER_OP("Warp")
		.Input("x: T")
    .Input("u: T")
		.Output("warped_x: T")
		.Attr("T: {float32, float64}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("WarpTranspose")
    .Input("grad_out: T")
    .Input("u: T")
		.Output("grad_x: T")
		.Attr("T: {float32, float64}")
  	.SetShapeFn(shape_inference::UnchangedShape);

template <typename T>
class TFWarpOperator : public OpKernel {
public:
	
	explicit TFWarpOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
  }

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& tf_x = context->input(0);
    const Tensor& tf_u = context->input(1);

    // Check dimensionality
    OP_REQUIRES(context, tf_x.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                      tf_x.dims(), "d."));
    OP_REQUIRES(context, tf_u.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                      tf_u.dims(), "d."));

    OP_REQUIRES(context, tf_u.dim_size(3) == 2,
                errors::Unimplemented("Expected the channel dimension to be 2, got ",
                                      tf_u.dim_size(3), "."));

		TensorShape output_shape = tf_x.shape();

		// allocate the output
		Tensor* tf_out = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &tf_out));

		// compute the output
		auto x = getDTensorTensorflow<T, 4>(tf_x);
    auto u = getDTensorTensorflow<T, 4>(tf_u);
		auto out = getDTensorTensorflow<T, 4>(*tf_out);
		
		optox::WarpOperator<T> op;
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({out.get()}, {x.get(), u.get()});
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Warp") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFWarpOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);


#undef REGISTER_GPU


template <typename T>
class TFWarpTransposeOperator : public OpKernel {
public:
	
	explicit TFWarpTransposeOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
    const Tensor& tf_grad_out = context->input(0);
    const Tensor& tf_u = context->input(1);

		TensorShape output_shape = tf_grad_out.shape();

    // Check dimensionality
    OP_REQUIRES(context, tf_u.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                      tf_u.dims(), "d."));
    OP_REQUIRES(context, tf_grad_out.dims() == 4,
                errors::Unimplemented("Expected a 4d Tensor, got ",
                                      tf_grad_out.dims(), "d."));
    OP_REQUIRES(context, tf_u.dim_size(3) == 2,
                errors::Unimplemented("Expected the channel dimension to be 2, got ",
                                      tf_u.dim_size(3), "."));

		// allocate the output
		Tensor* tf_grad_x = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, output_shape, &tf_grad_x));

		// compute the output
    auto grad_out = getDTensorTensorflow<T, 4>(tf_grad_out);
    auto u = getDTensorTensorflow<T, 4>(tf_u);
		auto grad_x = getDTensorTensorflow<T, 4>(*tf_grad_x);
		
		optox::WarpOperator<T> op;
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({grad_x.get()}, {grad_out.get(), u.get()});
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("WarpTranspose") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFWarpTransposeOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU