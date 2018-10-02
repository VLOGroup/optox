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

#include "tf_nabla_operator.h"

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("NablaOperator")
		.Input("x: T")
		.Output("nabla_x: T")
		.Attr("T: {float32, float64}")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
		TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(2), input, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("NablaOperatorAdjoint")
		.Input("y: T")
		.Output("div_y: T")
		.Attr("T: {float32, float64}")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
		// drop the first dimension
		TF_RETURN_IF_ERROR(c->Subshape(input, 1, 3, &output));
		c->set_output(0, output);
		return Status::OK();
	});

template <typename dtype>
class TFNablaOperator : public OpKernel {
public:
	
	explicit TFNablaOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape = x_tensor.shape();
      	out_shape.InsertDim(0, 2);

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		auto out_tt = output_tensor->tensor<dtype,3>();
        NablaOperatorForward<dtype>()(context->eigen_device<GPUDevice>(), out_tt, x_tensor.tensor<dtype,2>());
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("NablaOperator") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNablaOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename dtype>
class TFNablaOperatorAdjoint : public OpKernel {
public:
	
	explicit TFNablaOperatorAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape({x_tensor.shape().dim_size(1), x_tensor.shape().dim_size(2)});

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		auto out_tt = output_tensor->tensor<dtype,2>();
        NablaOperatorAdjoint<dtype>()(context->eigen_device<GPUDevice>(), out_tt, x_tensor.tensor<dtype,3>());
	}
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("NablaOperatorAdjoint") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNablaOperatorAdjoint<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
