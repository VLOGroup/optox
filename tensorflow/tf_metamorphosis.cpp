// tf_metamorphosis.cpp
#define EIGEN_USE_THREADS
#include "tf_metamorphosis.h"
#include "tf_activations.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Operator registration

// radial basis activation -----------------------------------------------------
REGISTER_OP("MetamorphosisWarp")
.Attr("T: realnumbertype")
.Input("x: T")
.Input("phi: T")
.Output("output: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'} = 'BILINEAR'")
.SetShapeFn(shape_inference::UnchangedShape)
.Doc(R"doc(
		perform interpolation of c volume according to displacements phi
  c_int = MetamorphosisWarp(x, phi)
)doc");

REGISTER_OP("MetamorphosisWarpGrad")
.Attr("T: realnumbertype")
.Input("x: T")
.Input("phi: T")
.Input("grad_out: T")
.Output("grad_x: T")
.Output("grad_phi: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'} = 'BILINEAR'")
.SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle x, phi;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &phi));
    c->set_output(0, x);
    c->set_output(1, phi);
    return Status::OK();
  })
.Doc(R"doc(
Gradient operator for metamorphosis warp.
grad_x, grad_phi = MetamorphosisWarpGrad(x, phi, grad_out)
)doc");


REGISTER_OP("Warp")
.Attr("T: realnumbertype")
.Input("x: T")
.Input("phi: T")
.Output("output: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'} = 'BILINEAR'")
.SetShapeFn(shape_inference::UnchangedShape)
.Doc(R"doc(
		perform interpolation of c volume according to displacements phi
  c_int = Warp(x, phi)
)doc");

REGISTER_OP("WarpGrad")
.Attr("T: realnumbertype")
.Input("x: T")
.Input("phi: T")
.Input("grad_out: T")
.Output("grad_x: T")
.Output("grad_phi: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'} = 'BILINEAR'")
.SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle x, phi;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &x));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &phi));
    c->set_output(0, x);
    c->set_output(1, phi);
    return Status::OK();
  })
.Doc(R"doc(
Gradient operator for metamorphosis warp.
grad_x, grad_phi = WarpGrad(x, phi, grad_out)
)doc");

template<typename Device, typename T>
class MetamorphosisWarpOp : public OpKernel {
  public:
    explicit MetamorphosisWarpOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& phi_tensor = context->input(1);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, x_tensor.dims() == 5,
                  errors::Unimplemented("Expected a 5d input Tensor, got ",
                                        x_tensor.dims(), "d."));
      OP_REQUIRES(context, phi_tensor.dims() == 4,
        errors::Unimplemented("Expected a 4d phi Tensor, got ",
                              phi_tensor.dims(), "d."));

      // Check whether the size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(3) == 3,
        errors::Unimplemented("Input tensor 4-th dimension must be 3!"));

      // Check whether size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(0) == x_tensor.shape().dim_size(0) &&
    		  	  phi_tensor.shape().dim_size(1) == x_tensor.shape().dim_size(3) &&
				      phi_tensor.shape().dim_size(2) == x_tensor.shape().dim_size(4),
              errors::Unimplemented("The inputs do not match! Expected x: SCRMN, phi: SMN3!"));

      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(),
                                                       &output_tensor));

      // Do the computation
      auto out_tensor_map = output_tensor->tensor<T,5>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
        	MetamorphosisInterpolationFunctor<Device, T, tficg::INTERPOLATE_BILINEAR>()(context,
                  x_tensor.tensor<T,5>(),
                  phi_tensor.tensor<T,4>(),
                  out_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
        	MetamorphosisInterpolationFunctor<Device, T, tficg::INTERPOLATE_BICUBIC>()(context,
        					x_tensor.tensor<T,5>(),
        					phi_tensor.tensor<T,4>(),
        					out_tensor_map);
          break;
        case tficg::INTERPOLATE_INVALID:
          break;
      }

    }

  private:
    tficg::interpolation_t interpolation_;
};

template<typename Device, typename T>
class MetamorphosisWarpGradOp : public OpKernel {
  public:
    explicit MetamorphosisWarpGradOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& phi_tensor = context->input(1);
      const Tensor& grad_out_tensor = context->input(2);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, x_tensor.dims() == 5,
                  errors::Unimplemented("Expected a 5d input Tensor, got ",
                                        x_tensor.dims(), "d."));
      OP_REQUIRES(context, phi_tensor.dims() == 4,
        errors::Unimplemented("Expected a 4d phi Tensor, got ",
                              phi_tensor.dims(), "d."));

      // Check whether the size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(3) == 3,
        errors::Unimplemented("Input tensor 4-th dimension must be 3!"));

      // Check whether size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(0) == x_tensor.shape().dim_size(0) &&
    		  	  phi_tensor.shape().dim_size(1) == x_tensor.shape().dim_size(3) &&
				      phi_tensor.shape().dim_size(2) == x_tensor.shape().dim_size(4),
              errors::Unimplemented("The inputs do not match! Expected x: SCRMN, phi: SMN3!"));

      // Check whether out_grad size matches the input
      OP_REQUIRES(context, x_tensor.shape() == grad_out_tensor.shape(),
                    errors::Unimplemented("The size of input x and grad_out must be identical"));

      // Create an output tensors
      Tensor *grad_x_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(),
    		  &grad_x_tensor));
      Tensor *grad_phi_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, phi_tensor.shape(),
    		  &grad_phi_tensor));

      // Do the computation
      auto grad_x_tensor_map = grad_x_tensor->tensor<T,5>();
      auto grad_phi_tensor_map = grad_phi_tensor->tensor<T,4>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
        	MetamorphosisInterpolationGradFunctor<Device, T, tficg::INTERPOLATE_BILINEAR>()(context,
                  x_tensor.tensor<T,5>(),
                  phi_tensor.tensor<T,4>(),
                  grad_out_tensor.tensor<T,5>(),
                  grad_x_tensor_map,
                  grad_phi_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
        	MetamorphosisInterpolationGradFunctor<Device, T, tficg::INTERPOLATE_BICUBIC>()(context,
        					x_tensor.tensor<T,5>(),
        					phi_tensor.tensor<T,4>(),
                  grad_out_tensor.tensor<T,5>(),
                  grad_x_tensor_map,
                  grad_phi_tensor_map);
          break;
        case tficg::INTERPOLATE_INVALID:
          break;
      }

    }

  private:
    tficg::interpolation_t interpolation_;
};

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("MetamorphosisWarp") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
	MetamorphosisWarpOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("MetamorphosisWarpGrad") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
	MetamorphosisWarpGradOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

// -----------------------------------------------------------------------------

template<typename Device, typename T>
class WarpOp : public OpKernel {
  public:
    explicit WarpOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& phi_tensor = context->input(1);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, x_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d input Tensor, got ",
                                        x_tensor.dims(), "d."));
      OP_REQUIRES(context, phi_tensor.dims() == 4,
        errors::Unimplemented("Expected a 4d phi Tensor, got ",
                              phi_tensor.dims(), "d."));

      // Check whether the size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(3) == 2,
        errors::Unimplemented("Input tensor 4-th dimension must be 2!"));

      // Check whether size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(0) == x_tensor.shape().dim_size(0) &&
    		  	  phi_tensor.shape().dim_size(1) == x_tensor.shape().dim_size(2) &&
				      phi_tensor.shape().dim_size(2) == x_tensor.shape().dim_size(3),
              errors::Unimplemented("The inputs do not match! Expected x: SCMN, phi: SMN2!"));

      // Create an output tensor
      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(),
                                                       &output_tensor));

      // Do the computation
      auto out_tensor_map = output_tensor->tensor<T,4>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
        	InterpolationFunctor<Device, T, tficg::INTERPOLATE_BILINEAR>()(context,
                  x_tensor.tensor<T,4>(),
                  phi_tensor.tensor<T,4>(),
                  out_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
        	InterpolationFunctor<Device, T, tficg::INTERPOLATE_BICUBIC>()(context,
        					x_tensor.tensor<T,4>(),
        					phi_tensor.tensor<T,4>(),
        					out_tensor_map);
          break;
        case tficg::INTERPOLATE_INVALID:
          break;
      }

    }

  private:
    tficg::interpolation_t interpolation_;
};


template<typename Device, typename T>
class WarpGradOp : public OpKernel {
  public:
    explicit WarpGradOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& phi_tensor = context->input(1);
      const Tensor& grad_out_tensor = context->input(2);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, x_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d input Tensor, got ",
                                        x_tensor.dims(), "d."));
      OP_REQUIRES(context, phi_tensor.dims() == 4,
        errors::Unimplemented("Expected a 4d phi Tensor, got ",
                              phi_tensor.dims(), "d."));

      // Check whether the size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(3) == 2,
        errors::Unimplemented("Input tensor 4-th dimension must be 2!"));

      // Check whether size fits
      OP_REQUIRES(context, phi_tensor.shape().dim_size(0) == x_tensor.shape().dim_size(0) &&
    		  	  phi_tensor.shape().dim_size(1) == x_tensor.shape().dim_size(2) &&
				      phi_tensor.shape().dim_size(2) == x_tensor.shape().dim_size(3),
              errors::Unimplemented("The inputs do not match! Expected x: SCMN, phi: SMN2!"));

      // Check whether out_grad size matches the input
      OP_REQUIRES(context, x_tensor.shape() == grad_out_tensor.shape(),
                    errors::Unimplemented("The size of input x and grad_out must be identical"));

      // Create an output tensors
      Tensor *grad_x_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, x_tensor.shape(),
    		  &grad_x_tensor));
      Tensor *grad_phi_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, phi_tensor.shape(),
    		  &grad_phi_tensor));

      // Do the computation
      auto grad_x_tensor_map = grad_x_tensor->tensor<T,4>();
      auto grad_phi_tensor_map = grad_phi_tensor->tensor<T,4>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
        	InterpolationGradFunctor<Device, T, tficg::INTERPOLATE_BILINEAR>()(context,
                  x_tensor.tensor<T,4>(),
                  phi_tensor.tensor<T,4>(),
                  grad_out_tensor.tensor<T,4>(),
                  grad_x_tensor_map,
                  grad_phi_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
        	InterpolationGradFunctor<Device, T, tficg::INTERPOLATE_BICUBIC>()(context,
        					x_tensor.tensor<T,4>(),
        					phi_tensor.tensor<T,4>(),
                  grad_out_tensor.tensor<T,4>(),
                  grad_x_tensor_map,
                  grad_phi_tensor_map);
          break;
        case tficg::INTERPOLATE_INVALID:
          break;
      }

    }

  private:
    tficg::interpolation_t interpolation_;
};

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("Warp") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
	WarpOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("WarpGrad") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
	WarpGradOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif
