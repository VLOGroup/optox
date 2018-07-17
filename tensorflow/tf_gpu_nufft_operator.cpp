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
#include "tensorflow/core/util/padding.h"

#include "tf_gpu_nufft_operator.h"
#include <gpuNUFFT_operator_factory.hpp>

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("GpuNufftForward")
		.Input("image: complex64")
		.Input("sensitivities: complex64")
		.Input("trajectory: float32")
		.Input("dcf: float32")
		.Output("rawdata: complex64")
		.Attr("osf: int >= 1")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("GpuNufftAdjoint")
		.Input("rawdata: complex64")
		.Input("sensitivities: complex64")
		.Input("trajectory: float32")
		.Input("dcf: float32")
		.Output("image: complex64")
		.Attr("osf: int >= 1")
		.Attr("sector_width: int >= 1")
		.Attr("kernel_width: int >= 1")
		.Attr("img_dim: int >= 1")
		.SetShapeFn(shape_inference::UnchangedShape);

class TFGpuNufftForwardOp : public OpKernel {
public:
	
	explicit TFGpuNufftForwardOp(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensors
		const Tensor& img_tensor = context->input(0);
		const Tensor& sensitivities_tensor = context->input(1);
		const Tensor& trajectory_tensor = context->input(2);
		const Tensor& dcf_tensor = context->input(3);

		// TODO size checks, dim checks!

		// Reshape tensors
		auto img = img_tensor.flat_inner_dims<complex64, 3>();
		auto sensitivities = sensitivities_tensor.flat_inner_dims<complex64, 4>();
		auto trajectory = trajectory_tensor.flat_inner_dims<float, 3>();
		auto dcf = dcf_tensor.flat_inner_dims<float, 2>();

		// Prepare output shape
		auto rawdata_shape = trajectory_tensor.shape();
		auto dims = trajectory_tensor.dims();
		rawdata_shape.set_dim(dims-2, sensitivities.dimensions()[1]);

		// Allocate the output
		Tensor* rawdata_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, rawdata_shape, &rawdata_tensor));

		// Reshape the output
		auto rawdata = rawdata_tensor->flat_inner_dims<complex64, 3>();

		applyGpuNufftForwardOperator()(context->eigen_device<GPUDevice>(),
		                               rawdata,
									   img,
									   sensitivities,
									   trajectory,
									   dcf,
									   osf_,
									   sector_width_,
									   kernel_width_,
									   img_dim_
									  );
	}

protected:
	int osf_;
	int sector_width_;
	int kernel_width_;
	int img_dim_;
};

class TFGpuNufftAdjointOp : public OpKernel {
public:
	
	explicit TFGpuNufftAdjointOp(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		// Get attributes
		OP_REQUIRES_OK(context, context->GetAttr("osf", &osf_));
		OP_REQUIRES_OK(context, context->GetAttr("sector_width", &sector_width_));
		OP_REQUIRES_OK(context, context->GetAttr("kernel_width", &kernel_width_));
		OP_REQUIRES_OK(context, context->GetAttr("img_dim", &img_dim_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensors
		const Tensor& rawdata_tensor = context->input(0);
		const Tensor& sensitivities_tensor = context->input(1);
		const Tensor& trajectory_tensor = context->input(2);
		const Tensor& dcf_tensor = context->input(3);

		// TODO size checks, dim checks!

		// Reshape tensors
		auto rawdata = rawdata_tensor.flat_inner_dims<complex64, 3>();
		auto sensitivities = sensitivities_tensor.flat_inner_dims<complex64, 4>();
		auto trajectory = trajectory_tensor.flat_inner_dims<float, 3>();
		auto dcf = dcf_tensor.flat_inner_dims<float, 2>();

		// Prepare output shape
		auto img_shape = rawdata_tensor.shape();
		auto dims = rawdata_tensor.dims();
		img_shape.set_dim(dims-2, img_dim_);
		img_shape.set_dim(dims-1, img_dim_);

		// Allocate the output
		Tensor* img_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, img_shape, &img_tensor));

		// Reshape the output
		auto img = img_tensor->flat_inner_dims<complex64, 3>();

		applyGpuNufftAdjointOperator()(context->eigen_device<GPUDevice>(),
		                               img,
									   rawdata,
									   sensitivities,
									   trajectory,
									   dcf,
									   osf_,
									   sector_width_,
									   kernel_width_,
									   img_dim_
									  );
	}

protected:
	int osf_;
	int sector_width_;
	int kernel_width_;
	int img_dim_;
};

REGISTER_KERNEL_BUILDER( \
		Name("GpuNufftForward") \
		.Device(DEVICE_GPU),
		TFGpuNufftForwardOp);

REGISTER_KERNEL_BUILDER( \
		Name("GpuNufftAdjoint") \
		.Device(DEVICE_GPU),
		TFGpuNufftAdjointOp);