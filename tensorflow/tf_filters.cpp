// tf_filters.cpp
#define EIGEN_USE_THREADS
#include "tf_filters.h"
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
REGISTER_OP("RotateFilter")
.Attr("T: realnumbertype")
.Input("x: T")
.Input("angles: T")
.Output("output: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'}")
.SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input;
    shape_inference::ShapeHandle angles;
    shape_inference::ShapeHandle output;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &angles));
    TF_RETURN_IF_ERROR(c->Concatenate(input, angles, &output));
    c->set_output(0, output);
    return Status::OK();
  })
    .Doc(R"doc(
Perform rotation of filters.
  output = RotateFilter(filter, angles)
)doc");

REGISTER_OP("RotateFilterGrad")
.Attr("T: realnumbertype")
.Input("angles: T")
.Input("grad_out: T")
.Output("grad_x: T")
.Attr("interpolation: {'BILINEAR', 'BICUBIC'}")
.SetShapeFn([](shape_inference::InferenceContext *c) {
    shape_inference::ShapeHandle input;
    shape_inference::ShapeHandle output;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &input));
    // drop the last dimension
    TF_RETURN_IF_ERROR(c->Subshape(input, 0, 4, &output));
    c->set_output(0, output);
    return Status::OK();
  })
.Doc(R"doc(
Perform rotation of filters.
output = RotateFilterGrad(angles, grad_out)
)doc");


// Activation base class -------------------------------------------------------
//  Defines the interface as ensures proper handinling of the in- and outputs
template<typename Device, typename T>
class RotateFilterOp : public OpKernel {
  public:
    explicit RotateFilterOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    virtual ~RotateFilterOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& x_tensor = context->input(0);
      const Tensor& angle_tensor = context->input(1);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, x_tensor.dims() == 4,
                  errors::Unimplemented("Expected a 4d Tensor, got ",
                                        x_tensor.dims(), "d."));
      OP_REQUIRES(context, angle_tensor.dims() == 1,
        errors::Unimplemented("Expected a 1d angle Tensor, got ",
                              angle_tensor.dims(), "d."));

      // Check whether the size fits
      auto x_shape = x_tensor.shape();
      OP_REQUIRES(context, x_shape.dim_size(0)*x_shape.dim_size(1) <= 1024,
        errors::Unimplemented("number of 2d filter taps of single channel too large!"));

      // Create an output tensor
      auto angle_shape = angle_tensor.shape();
      TensorShape out_shape = x_shape;
      out_shape.InsertDim(4, angle_shape.dim_size(0));

      Tensor *output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                       &output_tensor));

      // Do the computation
      auto out_tensor_map = output_tensor->tensor<T,5>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
          RotateFilterFunctor<Device, T, tficg::INTERPOLATE_BILINEAR>()(context,
            x_tensor.tensor<T,4>(),
            angle_tensor.tensor<T,1>(),
            out_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
          RotateFilterFunctor<Device, T, tficg::INTERPOLATE_BICUBIC>()(context,
            x_tensor.tensor<T,4>(),
            angle_tensor.tensor<T,1>(),
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
class RotateFilterGradOp : public OpKernel {
  public:
    explicit RotateFilterGradOp(OpKernelConstruction* context) : OpKernel(context)
    {
      std::string interpolation_str;
      OP_REQUIRES_OK(context, context->GetAttr("interpolation", &interpolation_str));
      interpolation_ = tficg::strToInterpolation(interpolation_str);
      OP_REQUIRES(context, interpolation_ != tficg::INTERPOLATE_INVALID,
        errors::Unimplemented("Not supported INTERPOLATION type!"));
    }

    virtual ~RotateFilterGradOp() {};

    void Compute(OpKernelContext* context) override
    {
      // Grab the input tensors
      const Tensor& angle_tensor = context->input(0);
      const Tensor& grad_out_tensor = context->input(1);

      // Check the dimensionality and size of the filters and angles
      OP_REQUIRES(context, grad_out_tensor.dims() == 5,
                  errors::Unimplemented("Expected a 5d Tensor, got ",
                  grad_out_tensor.dims(), "d."));
      OP_REQUIRES(context, angle_tensor.dims() == 1,
        errors::Unimplemented("Expected a 1d angle Tensor, got ",
                              angle_tensor.dims(), "d."));

      // Check whether the size fits
      auto grad_out_shape = grad_out_tensor.shape();
      OP_REQUIRES(context, grad_out_shape.dim_size(0)*grad_out_shape.dim_size(1) <= 1024,
        errors::Unimplemented("number of 2d filter taps of single channel too large!"));

      // Create an output tensor
      TensorShape grad_x_shape = grad_out_shape;
      grad_x_shape.RemoveDim(4);

      Tensor *grad_x_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, grad_x_shape,
                                                       &grad_x_tensor));

      // Do the computation
      auto grad_x_tensor_map = grad_x_tensor->tensor<T,4>();
      switch (interpolation_)
      {
        case tficg::INTERPOLATE_BILINEAR:
          RotateFilterGradFunctor<Device, T,tficg::INTERPOLATE_BILINEAR>()(context,
            angle_tensor.tensor<T,1>(),
            grad_out_tensor.tensor<T,5>(),
            grad_x_tensor_map);
          break;
        case tficg::INTERPOLATE_BICUBIC:
          RotateFilterGradFunctor<Device, T,tficg::INTERPOLATE_BICUBIC>()(context,
            angle_tensor.tensor<T,1>(),
            grad_out_tensor.tensor<T,5>(),
            grad_x_tensor_map);
          break;
        case tficg::INTERPOLATE_INVALID:
          break;
      }
    }

  private:
    tficg::interpolation_t interpolation_;
};

// -----------------------------------------------------------------------------

// Radial basis function activation --------------------------------------------

// template <typename T, tficg::interpolation_t I>
// struct RotateFilterFunctor<CPUDevice, T, I> {
//   void operator()(OpKernelContext *context,
//                   const typename Tensor4<T>::ConstTensor &x,
//                   const typename Tensor1<T>::ConstTensor &angles,
//                   typename Tensor5<T>::Tensor &out) {
//     // TODO: implement CPU functor
//     std::cout << "Using rotate filter CPU kernel!" << std::endl;
//   }
// };

// #define REGISTER_CPU(T) \
// REGISTER_KERNEL_BUILDER(  \
//     Name("RotateFilter") \
//     .Device(DEVICE_CPU) \
//     .TypeConstraint<T>("T"), \
//     RotateFilterOp<CPUDevice, T>);

// TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("RotateFilter") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    RotateFilterOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif


// template <typename T, tficg::interpolation_t I>
// struct RotateFilterGradFunctor<CPUDevice, T, I> {
//   void operator()(OpKernelContext* context,
//                   const typename Tensor1<T>::ConstTensor &angles,
//                   const typename Tensor5<T>::ConstTensor &grad_out,
//                   typename Tensor4<T>::Tensor &grad_x) {
//     // TODO: implement CPU functor
//     std::cout << "Using CPU rotate filter gradient kernel!" << std::endl;
//     // initialize the gradient w
//     grad_x.setZero();
//   }
// };

// #define REGISTER_CPU(T) \
// REGISTER_KERNEL_BUILDER(  \
//     Name("RotateFilterGrad") \
//     .Device(DEVICE_CPU) \
//     .TypeConstraint<T>("T"), \
//     RotateFilterGradOp<CPUDevice, T>);

// TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

#if GOOGLE_CUDA
#define REGISTER_GPU(T) \
REGISTER_KERNEL_BUILDER(  \
    Name("RotateFilterGrad") \
    .Device(DEVICE_GPU) \
    .TypeConstraint<T>("T"), \
    RotateFilterGradOp<GPUDevice, T>) \

TF_CALL_ICG_REAL_NUMBER_TYPES(REGISTER_GPU);
#undef REGISTER_GPU
#endif

// -----------------------------------------------------------------------------
