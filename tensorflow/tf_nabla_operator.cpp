///@file th_nabla_operator.h
///@brief PyTorch wrappers for nabla operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019

#include <vector>

#include "tf_utils.h"
#include "operators/nabla_operator.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include "tensorflow/core/framework/register_types.h"
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>


using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("Nabla2dOperator")
		.Input("x: T")
		.Output("nabla_x: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
		TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(2), input, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("Nabla3dOperator")
		.Input("x: T")
		.Output("nabla_x: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.Attr("hz: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
		TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(3), input, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("Nabla4dOperator")
		.Input("x: T")
		.Output("nabla_x: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.Attr("hz: float = 1.0")
		.Attr("ht: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
		TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(4), input, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("Nabla2dOperatorAdjoint")
		.Input("y: T")
		.Output("div_y: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
		// drop the first dimension
		TF_RETURN_IF_ERROR(c->Subshape(input, 1, 3, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("Nabla3dOperatorAdjoint")
		.Input("y: T")
		.Output("div_y: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.Attr("hz: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
		// drop the first dimension
		TF_RETURN_IF_ERROR(c->Subshape(input, 1, 4, &output));
		c->set_output(0, output);
		return Status::OK();
	});

REGISTER_OP("Nabla4dOperatorAdjoint")
		.Input("y: T")
		.Output("div_y: T")
		.Attr("T: {float32, float64}")
		.Attr("hx: float = 1.0")
		.Attr("hy: float = 1.0")
		.Attr("hz: float = 1.0")
		.Attr("ht: float = 1.0")
		.SetShapeFn([](shape_inference::InferenceContext *c) {
		shape_inference::ShapeHandle input;
		shape_inference::ShapeHandle output;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));
		// drop the first dimension
		TF_RETURN_IF_ERROR(c->Subshape(input, 1, 5, &output));
		c->set_output(0, output);
		return Status::OK();
	});
template <typename dtype>
class TFNabla2dOperator : public OpKernel {
public:
	
	explicit TFNabla2dOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape = x_tensor.shape();
      	out_shape.InsertDim(0, 2);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 2>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 3>(*output_tensor);
		
		optox::NablaOperator<dtype, 2> op(hx_, hy_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla2dOperator") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla2dOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename dtype>
class TFNabla3dOperator : public OpKernel {
public:
	
	explicit TFNabla3dOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	    OP_REQUIRES_OK(context, context->GetAttr("hz", &hz_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape = x_tensor.shape();
      	out_shape.InsertDim(0, 3);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 3>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 4>(*output_tensor);
		
		optox::NablaOperator<dtype, 3> op(hx_, hy_, hz_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
		float hz_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla3dOperator") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla3dOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename dtype>
class TFNabla4dOperator : public OpKernel {
public:
	
	explicit TFNabla4dOperator(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	    OP_REQUIRES_OK(context, context->GetAttr("hz", &hz_));
		OP_REQUIRES_OK(context, context->GetAttr("ht", &ht_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape = x_tensor.shape();
      	out_shape.InsertDim(0, 4);

		// allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 4>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 5>(*output_tensor);
		
		optox::NablaOperator<dtype, 4> op(hx_, hy_, hz_, ht_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.forward({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
		float hz_;
		float ht_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla4dOperator") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla4dOperator<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
template <typename dtype>
class TFNabla2dOperatorAdjoint : public OpKernel {
public:
	
	explicit TFNabla2dOperatorAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape({x_tensor.shape().dim_size(1), x_tensor.shape().dim_size(2)});

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 3>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 2>(*output_tensor);
		
		optox::NablaOperator<dtype, 2> op(hx_, hy_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla2dOperatorAdjoint") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla2dOperatorAdjoint<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU


#undef REGISTER_GPU
template <typename dtype>
class TFNabla3dOperatorAdjoint : public OpKernel {
public:
	
	explicit TFNabla3dOperatorAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	    OP_REQUIRES_OK(context, context->GetAttr("hz", &hz_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape({x_tensor.shape().dim_size(1), x_tensor.shape().dim_size(2), x_tensor.shape().dim_size(3)});

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 4>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 3>(*output_tensor);
		
		optox::NablaOperator<dtype, 3> op(hx_, hy_, hz_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
		float hz_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla3dOperatorAdjoint") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla3dOperatorAdjoint<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename dtype>
class TFNabla4dOperatorAdjoint : public OpKernel {
public:
	
	explicit TFNabla4dOperatorAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	    OP_REQUIRES_OK(context, context->GetAttr("hx", &hx_));
	    OP_REQUIRES_OK(context, context->GetAttr("hy", &hy_));
	    OP_REQUIRES_OK(context, context->GetAttr("hz", &hz_));
		OP_REQUIRES_OK(context, context->GetAttr("ht", &ht_));
	}

	void Compute(OpKernelContext* context) override {
		// Grab the input tensor
		const Tensor& x_tensor = context->input(0);

		TensorShape out_shape({x_tensor.shape().dim_size(1), x_tensor.shape().dim_size(2), x_tensor.shape().dim_size(3), x_tensor.shape().dim_size(4)});

		//allocate the output
		Tensor* output_tensor = nullptr;
		OP_REQUIRES_OK(context,
			context->allocate_output(0, out_shape, &output_tensor));

		// compute the output
		auto input = getDTensorTensorflow<dtype, 5>(x_tensor);
		auto output = getDTensorTensorflow<dtype, 4>(*output_tensor);
		
		optox::NablaOperator<dtype, 4> op(hx_, hy_, hz_, ht_);
		op.setStream(context->eigen_device<GPUDevice>().stream());
		op.adjoint({output.get()}, {input.get()});
	}

	private:
		float hx_;
		float hy_;
		float hz_;
		float ht_;
};

#define REGISTER_GPU(type) \
	REGISTER_KERNEL_BUILDER( \
		Name("Nabla4dOperatorAdjoint") \
		.Device(DEVICE_GPU) \
		.TypeConstraint<type>("T"), \
		TFNabla4dOperatorAdjoint<type>) \

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
