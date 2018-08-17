///@file tfaddoperator.cpp
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include <iostream>
#include <cuda.h>


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tf_add_operator.h"

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("AddOperator")
		.Input("a: T")
		.Input("b: T")
		.Output("c: T")
		.Attr("T: {float32, float64}")
		.SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("AddOperatorAdjoint")
		.Input("a: T")
		.Output("b: T")
		.Output("c: T")
		.Attr("T: {float32, float64}")
		.SetShapeFn(shape_inference::UnchangedShape);

template <typename dtype>
class TFAddOperator : public OpKernel {
public:
	
	explicit TFAddOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& a_tensor = context->input(0);
		const Tensor& b_tensor = context->input(1);
		
		//flatten tensors
		auto a_flat = a_tensor.flat<dtype>();
		auto b_flat = b_tensor.flat<dtype>();

		OP_REQUIRES(context, a_tensor.shape() == b_tensor.shape(),
                  errors::Unimplemented("Invalid shape! ",
                                        a_tensor.shape(), ", != ", b_tensor.shape()));

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, a_tensor.shape(), &output_tensor));

		//get flat version to fill
		auto output = output_tensor->flat<dtype>();

        AddOperatorForward<dtype>()(context->eigen_device<GPUDevice>(), output, a_flat, b_flat);
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("AddOperator") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFAddOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename dtype>
class TFAddOperatorAdjoint : public OpKernel {
public:
	
	explicit TFAddOperatorAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& a_tensor = context->input(0);
		
		//flatten tensors
		auto a_flat = a_tensor.flat<dtype>();

		//allocate the output
		Tensor* output_b_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, a_tensor.shape(), &output_b_tensor));
		Tensor* output_c_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, a_tensor.shape(), &output_c_tensor));

		//get flat version to fill
		auto output_b = output_b_tensor->flat<dtype>();
		auto output_c = output_c_tensor->flat<dtype>();

        AddOperatorAdjoint<dtype>()(context->eigen_device<GPUDevice>(), output_b, output_c, a_flat);
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("AddOperatorAdjoint") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFAddOperatorAdjoint<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
