///@file tfaddoperator.cpp
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include <iostream>
#include <cuda.h>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"

#include <iu/iucore/lineardevicememory.h>

#include "tf_nabla_operator.h"
#include "operators/nabla_operator.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void NablaOperatorForward<T>::operator()(const GPUDevice& d, 
	  typename Tensor3<T>::Tensor &out, 
	  const typename Tensor2<T>::ConstTensor &x)
{
	iu::Size<2> input_size({x.dimensions()[1], x.dimensions()[0]});
	iu::Size<3> output_size({out.dimensions()[2], out.dimensions()[1], out.dimensions()[0]});

	iu::LinearDeviceMemory<T, 2> iu_in(const_cast<T*>(x.data()), input_size, true);

	iu::LinearDeviceMemory<T, 3> iu_out(out.data(), output_size, true);

	optox::NablaOperator<T, 2> op;
	op.setStream(d.stream());
	op.forward({&iu_out}, {&iu_in});
}

#define REGISTER_GPU(type) \
	template class NablaOperatorForward<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
void NablaOperatorAdjoint<T>::operator()(const GPUDevice& d, 
	  typename Tensor2<T>::Tensor &out, 
	  const typename Tensor3<T>::ConstTensor &x)
{
	iu::Size<3> input_size({x.dimensions()[2], x.dimensions()[1], x.dimensions()[0]});
	iu::Size<2> output_size({out.dimensions()[1], out.dimensions()[0]});

	iu::LinearDeviceMemory<T, 3> iu_in(const_cast<T*>(x.data()), input_size, true);

	iu::LinearDeviceMemory<T, 2> iu_out(out.data(), output_size, true);
	
	optox::NablaOperator<T, 2> op;
	op.setStream(d.stream());
	op.adjoint({&iu_out}, {&iu_in});
}

#define REGISTER_GPU(type) \
	template class NablaOperatorAdjoint<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
