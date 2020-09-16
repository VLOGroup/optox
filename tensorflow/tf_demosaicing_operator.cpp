///@file tf_demosiaing.cpp
///@brief tensorflow wrappers for the demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 09.07.2018

#include "tf_utils.h"
#include "operators/demosaicing_operator.h"

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/common_shape_fns.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/platform/default/integral_types.h>
#include <tensorflow/core/util/tensor_format.h>

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/**
 * register the operation with necessary options
 */
REGISTER_OP("DemosaicingOperatorForward")
    .Input("input: T")
    .Output("output: T")
    .Attr("bayerpattern: {'RGGB', 'BGGR', 'GBRG', 'GRBG'} = 'RGGB'")
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
    .Output("output: T")
    .Attr("bayerpattern: {'RGGB', 'BGGR', 'GBRG', 'GRBG'} = 'RGGB'")
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
class TFDemosaicingForward : public OpKernel
{
  public:
    explicit TFDemosaicingForward(OpKernelConstruction *context)
        : OpKernel(context)
    {
        //Check any attributes
        std::string pattern_str;
        OP_REQUIRES_OK(context, context->GetAttr("bayerpattern", &pattern_str));

        op_ = new optox::DemosaicingOperator<T>(pattern_str);
    }

    virtual ~TFDemosaicingForward()
    {
        if (op_)
            delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);

        // Check dimensionality
        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::Unimplemented("Expected a 4d Tensor, got ",
                                          input_tensor.dims(), "d."));

        OP_REQUIRES(context, input_tensor.dim_size(3) == 3,
                    errors::Unimplemented("Expected the channel dimension to be 3, got ",
                                          input_tensor.dim_size(3), "."));

        // Prepare output shape
        auto output_shape = input_tensor.shape();
        output_shape.set_dim(3, 1);

        // Create an output tensor
        Tensor *output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output_tensor));

        // compute the output
        auto input = getDTensorTensorflow<T, 4>(input_tensor);
        auto output = getDTensorTensorflow<T, 4>(*output_tensor);

        op_->setStream(context->eigen_device<GPUDevice>().stream());
        op_->forward({output.get()}, {input.get()});
    }

  private:
    optox::DemosaicingOperator<T> *op_ = nullptr;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("DemosaicingOperatorForward") \
            .Device(DEVICE_GPU)            \
            .HostMemory("pattern")         \
            .TypeConstraint<T>("T"),       \
        TFDemosaicingForward<T>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
class TFDemosaicingAdjoint : public OpKernel
{
  public:
    explicit TFDemosaicingAdjoint(OpKernelConstruction *context)
        : OpKernel(context)
    {
        //Check any attributes
        std::string pattern_str;
        OP_REQUIRES_OK(context, context->GetAttr("bayerpattern", &pattern_str));

        op_ = new optox::DemosaicingOperator<T>(pattern_str);
    }

    virtual ~TFDemosaicingAdjoint()
    {
        if (op_)
            delete op_;
    }

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);

        // Check dimensionality
        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::Unimplemented("Expected a 4d Tensor, got ",
                                          input_tensor.dims(), "d."));

        OP_REQUIRES(context, input_tensor.dim_size(3) == 1,
                    errors::Unimplemented("Expected the channel dimension to be 1, got ",
                                          input_tensor.dim_size(3), "."));

        // Prepare output shape
        auto output_shape = input_tensor.shape();

        output_shape.set_dim(3, 3);

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                         &output_tensor));

        // compute the output
        auto input = getDTensorTensorflow<T, 4>(input_tensor);
        auto output = getDTensorTensorflow<T, 4>(*output_tensor);

        op_->setStream(context->eigen_device<GPUDevice>().stream());
        op_->adjoint({output.get()}, {input.get()});
    }

  private:
    optox::DemosaicingOperator<T> *op_ = nullptr;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("DemosaicingOperatorAdjoint") \
            .Device(DEVICE_GPU)            \
            .HostMemory("pattern")         \
            .TypeConstraint<T>("T"),       \
        TFDemosaicingAdjoint<T>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
