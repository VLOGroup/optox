///@file tf_demosiaing.cpp
///@brief tensorflow wrappers for the demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 09.07.2018

#include <iostream>
#include <cuda.h>


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tf_demosaicing.h"

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("DemosaicingOperatorForward")
    .Input("input: T")
    .Input("pattern: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
        shape_inference::ShapeHandle input_3;
        TF_RETURN_IF_ERROR(c->Subshape(input, 0, 3, &input_3));
        shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(input_3, c->Vector(1), &output));
        c->set_output(0, output);
        return Status::OK();
    });

REGISTER_OP("DemosaicingOperatorAdjoint")
    .Input("input: T")
    .Input("pattern: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
        shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
        shape_inference::ShapeHandle input_3;
        TF_RETURN_IF_ERROR(c->Subshape(input, 0, 3, &input_3));
        shape_inference::ShapeHandle output;
        TF_RETURN_IF_ERROR(c->Concatenate(input_3, c->Vector(3), &output));
        c->set_output(0, output);
        return Status::OK();
    });

template <typename T>
class TFDemosaicingForward : public OpKernel {
public:
	
	explicit TFDemosaicingForward(OpKernelConstruction* context) 
		: OpKernel(context)
	{
		//Check any attributes
	}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensor
        const Tensor& input_tensor = context->input(0);

        // Check dimensionality
        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::Unimplemented("Expected a 4d Tensor, got ",
                                            input_tensor.dims(), "d."));
        int N = input_tensor.dims();

        OP_REQUIRES(context, input_tensor.dim_size(3) == 3,
                    errors::Unimplemented("Expected the channel dimension to be 3, got ",
                                            input_tensor.dim_size(3), "."));

        // extract the bayer pattern
        const Tensor &pattern_tensor = context->input(1);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(pattern_tensor.shape()),
                    errors::Unimplemented("Expected pattern to be a scalar!"));

        int p = pattern_tensor.scalar<int>()();
        OP_REQUIRES(context, p >= 0 && p < 4,
                    errors::Unimplemented("Invalid pattern!"));
        optox::BayerPattern pattern = static_cast<optox::BayerPattern>(p);

        // Prepare output shape
        auto output_shape = input_tensor.shape();

        output_shape.set_dim(3, 1);

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                        &output_tensor));

        // Flat inner dimensions
        auto in = input_tensor.tensor<T,4>();
        auto out = output_tensor->tensor<T,4>();

        // Call the kernel
        DemosaicingOperatorWrapper<T>(pattern).forward(context->eigen_device<GPUDevice>(), out, in);
    }
};

#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER( \
		Name("DemosaicingOperatorForward") \
		.Device(DEVICE_GPU) \
        .HostMemory("pattern") \
		.TypeConstraint<T>("T"), \
		TFDemosaicingForward<T>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
class TFDemosaicingAdjoint : public OpKernel {
public:
	
	explicit TFDemosaicingAdjoint(OpKernelConstruction* context) 
		: OpKernel(context)
	{
	}

	void Compute(OpKernelContext* context) override
    {
	    // Grab the input tensor
        const Tensor& input_tensor = context->input(0);

        // Check dimensionality
        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::Unimplemented("Expected a 4d Tensor, got ",
                                            input_tensor.dims(), "d."));
        int N = input_tensor.dims();

        OP_REQUIRES(context, input_tensor.dim_size(3) == 1,
                    errors::Unimplemented("Expected the channel dimension to be 1, got ",
                                            input_tensor.dim_size(3), "."));

        // extract the bayer pattern
        const Tensor &pattern_tensor = context->input(1);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(pattern_tensor.shape()),
                    errors::Unimplemented("Expected pattern to be a scalar!"));

        int p = pattern_tensor.scalar<int>()();
        OP_REQUIRES(context, p >= 0 && p < 4,
                    errors::Unimplemented("Invalid pattern!"));
        optox::BayerPattern pattern = static_cast<optox::BayerPattern>(p);

        // Prepare output shape
        auto output_shape = input_tensor.shape();

        output_shape.set_dim(3, 3);

        // Create an output tensor
        Tensor* output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                        &output_tensor));

        // Flat inner dimensions
        auto in = input_tensor.tensor<T,4>();
        auto out = output_tensor->tensor<T,4>();

        // Call the kernel
        DemosaicingOperatorWrapper<T>(pattern).adjoint(context->eigen_device<GPUDevice>(), out, in);
	}
};

#define REGISTER_GPU(T) \
	REGISTER_KERNEL_BUILDER( \
		Name("DemosaicingOperatorAdjoint") \
		.Device(DEVICE_GPU) \
        .HostMemory("pattern") \
		.TypeConstraint<T>("T"), \
		TFDemosaicingAdjoint<T>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
