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

		// create NUFFT operator
		gpuNUFFT::Dimensions img_dims;
		img_dims.width = img_dim_;
		img_dims.height = img_dim_;
		img_dims.depth = 0;

		gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true);
		gpuNUFFT::GpuNUFFTOperator* nufft_op = nullptr;

		// loop over samples
		int samples = dcf.dimensions()[0];
		std::cout << "samples: " << samples << std::endl;

		gpuNUFFT::Array<float> trajectory_gpunufft;
		std::cout << "dcf length: " << dcf.dimensions()[1] << std::endl;
		trajectory_gpunufft.dim.length = dcf.dimensions()[1];
		int trajectory_offset = trajectory.size() / samples;

		gpuNUFFT::Array<float> dcf_gpunufft;
		dcf_gpunufft.dim.length = dcf.dimensions()[1];
		int dcf_offset = dcf.size() / samples;

		gpuNUFFT::Array<float2> sensitivities_gpunufft;
		sensitivities_gpunufft.dim = img_dims;
		std::cout << "nCoils: " << sensitivities.dimensions()[1] << std::endl;
		sensitivities_gpunufft.dim.channels = sensitivities.dimensions()[1];
		int sensitivities_offset = sensitivities.size() / samples;

		gpuNUFFT::Array<float2> img_gpunufft;
		img_gpunufft.dim = img_dims;
		int img_offset = img.size() / samples;

		gpuNUFFT::Array<float2> rawdata_gpunufft;
		std::cout << "rawdata length: " << rawdata.dimensions()[2] << std::endl;
		std::cout << "rawdata channels: " << rawdata.dimensions()[1] << std::endl;
		rawdata_gpunufft.dim.length = rawdata.dimensions()[2];
		rawdata_gpunufft.dim.channels = rawdata.dimensions()[1];
		int rawdata_offset = rawdata.size() / samples;

		std::cout << "trajectory_offset " << trajectory_offset << std::endl;
		std::cout << "dcf_offset " << dcf_offset << std::endl;
		std::cout << "sensitivities_offset " << sensitivities_offset << std::endl;
		std::cout << "rawdata_offset " << rawdata_offset << std::endl;
		std::cout << "image_offset " << img_offset << std::endl;

		for (int n = 0; n < samples; n++)
		{
			trajectory_gpunufft.data = trajectory.data() + n * trajectory_offset;
			dcf_gpunufft.data = dcf.data() + n * dcf_offset;
			sensitivities_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(sensitivities.data())) + n * sensitivities_offset;

			gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(trajectory_gpunufft,
													dcf_gpunufft, 
													sensitivities_gpunufft,
													kernel_width_,
													sector_width_,
													osf_,
													img_dims);
			
			img_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(img.data())) + n * img_offset;
			rawdata_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(rawdata.data())) + n * rawdata_offset;

			nufft_op->performForwardGpuNUFFT(img_gpunufft, rawdata_gpunufft);

			delete nufft_op;
		}
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

		// create NUFFT operator
		gpuNUFFT::Dimensions img_dims;
		img_dims.width = img_dim_;
		img_dims.height = img_dim_;
		img_dims.depth = 0;

		gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true);
		
		// loop over samples
		int samples = dcf.dimensions()[0];
		std::cout << "samples: " << samples << std::endl;

		gpuNUFFT::Array<float> trajectory_gpunufft;
		std::cout << "dcf length: " << dcf.dimensions()[1] << std::endl;
		trajectory_gpunufft.dim.length = dcf.dimensions()[1];
		int trajectory_offset = trajectory.size() / samples;

		gpuNUFFT::Array<float> dcf_gpunufft;
		dcf_gpunufft.dim.length = dcf.dimensions()[1];
		int dcf_offset = dcf.size() / samples;

		gpuNUFFT::Array<float2> sensitivities_gpunufft;
		sensitivities_gpunufft.dim = img_dims;
		std::cout << "nCoils: " << sensitivities.dimensions()[1] << std::endl;
		sensitivities_gpunufft.dim.channels = sensitivities.dimensions()[1];
		int sensitivities_offset = sensitivities.size() / samples;

		gpuNUFFT::Array<float2> img_gpunufft;
		img_gpunufft.dim = img_dims;
		int img_offset = img.size() / samples;

		gpuNUFFT::Array<float2> rawdata_gpunufft;
		std::cout << "rawdata length: " << rawdata.dimensions()[2] << std::endl;
		std::cout << "rawdata channels: " << rawdata.dimensions()[1] << std::endl;
		rawdata_gpunufft.dim.length = rawdata.dimensions()[2];
		rawdata_gpunufft.dim.channels = rawdata.dimensions()[1];
		int rawdata_offset = rawdata.size() / samples;

		std::cout << "trajectory_offset " << trajectory_offset << std::endl;
		std::cout << "dcf_offset " << dcf_offset << std::endl;
		std::cout << "sensitivities_offset " << sensitivities_offset << std::endl;
		std::cout << "rawdata_offset " << rawdata_offset << std::endl;
		std::cout << "img_offset " << img_offset << std::endl;

		for (int n = 0; n < samples; n++)
		{
			trajectory_gpunufft.data = trajectory.data() + n * trajectory_offset;
			dcf_gpunufft.data = dcf.data() + n * dcf_offset;
			sensitivities_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(sensitivities.data())) + n * sensitivities_offset;

			gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(trajectory_gpunufft,
													dcf_gpunufft, 
													sensitivities_gpunufft,
													kernel_width_,
													sector_width_,
													osf_,
													img_dims);
			
			img_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(img.data())) + n * img_offset;
			rawdata_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(rawdata.data())) + n * rawdata_offset;

			nufft_op->performGpuNUFFTAdj(rawdata_gpunufft, img_gpunufft);

			delete nufft_op;
		}
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