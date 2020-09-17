#include "tf_utils.h"

#include "operators/activations/act.h"
#include "operators/activations/act_rbf.h"
#include "operators/activations/act_linear.h"
#include "operators/activations/act_spline.h"

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

// Operator registration
REGISTER_OP("RbfAct")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RbfActGrad")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Input("grad_output: T")
    .Output("grad_in: T")
    .Output("grad_w: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("LinearAct")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("LinearActGrad")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Input("grad_output: T")
    .Output("grad_in: T")
    .Output("grad_w: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SplineAct")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Output("output: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("SplineActGrad")
    .Attr("T: realnumbertype")
    .Input("x: T")
    .Input("w: T")
    .Input("grad_output: T")
    .Output("grad_in: T")
    .Output("grad_w: T")
    .Attr("vmin: float")
    .Attr("vmax: float")
    .SetShapeFn(shape_inference::UnchangedShape);
/**
 * Activation operator interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename T, typename TOperator>
class ActivationBaseOp : public OpKernel {
  public:
    explicit ActivationBaseOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("vmin", &v_tmp));
      vmin_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("vmax", &v_tmp));
      vmax_ = static_cast<T>(v_tmp);

      op_ = new TOperator(vmin_, vmax_);
    }

    virtual ~ActivationBaseOp()
    {
        if (op_)
            delete op_;
    };

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& tf_input = context->input(0);
      const Tensor& tf_weights = context->input(1);

      // Create an output tensor
      Tensor *tf_output = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, tf_input.shape(),
                                                       &tf_output));
      // Do the computation
      auto input = getDTensorTensorflow<T, 2>(tf_input);
      auto weights = getDTensorTensorflow<T, 2>(tf_weights);
      auto output = getDTensorTensorflow<T, 2>(*tf_output);

      op_->forward({output.get()}, {input.get(), weights.get()});
    }

  protected:
    T vmin_, vmax_;
    TOperator *op_ = nullptr;
};

/**
 * Activation operator interface
 * Defines the IOs, attributes and performs size checks.
 */
template<typename T, typename TOperator>
class ActivationBaseGradOp : public OpKernel {
  public:
    explicit ActivationBaseGradOp(OpKernelConstruction* context) : OpKernel(context)
    {
      // Get attributes
      float v_tmp;
      OP_REQUIRES_OK(context, context->GetAttr("vmin", &v_tmp));
      vmin_ = static_cast<T>(v_tmp);
      OP_REQUIRES_OK(context, context->GetAttr("vmax", &v_tmp));
      vmax_ = static_cast<T>(v_tmp);

      op_ = new TOperator(vmin_, vmax_);
    }

    virtual ~ActivationBaseGradOp()
    {
        if (op_)
            delete op_;
    };

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& tf_input = context->input(0);
      const Tensor& tf_weights = context->input(1);
      const Tensor& tf_grad_out = context->input(2);

      // Create an output tensor
      Tensor *tf_grad_in = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, tf_input.shape(),
                                                       &tf_grad_in));
      Tensor *tf_grad_weights = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, tf_weights.shape(),
                                                       &tf_grad_weights));
      // Do the computation
      auto input = getDTensorTensorflow<T, 2>(tf_input);
      auto weights = getDTensorTensorflow<T, 2>(tf_weights);
      auto grad_out = getDTensorTensorflow<T, 2>(tf_grad_out);
      auto grad_in = getDTensorTensorflow<T, 2>(*tf_grad_in);
      auto grad_weights = getDTensorTensorflow<T, 2>(*tf_grad_weights);

      op_->adjoint({grad_in.get(), grad_weights.get()}, {input.get(), weights.get(), grad_out.get()});
    }

  protected:
    T vmin_, vmax_;
    TOperator *op_ = nullptr;
};

#define REGISTER_GPU(T)                    \
    REGISTER_KERNEL_BUILDER(               \
        Name("RbfAct") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseOp<T, optox::RBFActOperator<T>>) \
    REGISTER_KERNEL_BUILDER(               \
        Name("RbfActGrad") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseGradOp<T, optox::RBFActOperator<T>>) \
    REGISTER_KERNEL_BUILDER(               \
        Name("LinearAct") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseOp<T, optox::LinearActOperator<T>>) \
    REGISTER_KERNEL_BUILDER(               \
        Name("LinearActGrad") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseGradOp<T, optox::LinearActOperator<T>>) \
    REGISTER_KERNEL_BUILDER(               \
        Name("SplineAct") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseOp<T, optox::SplineActOperator<T>>) \
    REGISTER_KERNEL_BUILDER(               \
        Name("SplineActGrad") \
            .Device(DEVICE_GPU)            \
            .TypeConstraint<T>("T"),       \
        ActivationBaseGradOp<T, optox::SplineActOperator<T>>)

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
