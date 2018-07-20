///@file tfaddoperator.cpp
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include <iostream>
#include <cuda.h>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"

#include <iu/iucore/lineardevicememory.h>

#include "tf_add_operator.h"
#include "operators/add_operator.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void AddOperatorForward<T>::operator()(const GPUDevice& d, 
	typename Tensor1<T>::Tensor &output, 
	const typename Tensor1<T>::ConstTensor &a_flat, 
	const typename Tensor1<T>::ConstTensor &b_flat)
{
	iu::Size<1> size(a_flat.size());

	iu::LinearDeviceMemory<T, 1> iu_a(const_cast<T*>(a_flat.data()), size, true);
	iu::LinearDeviceMemory<T, 1> iu_b(const_cast<T*>(b_flat.data()), size, true);

	iu::LinearDeviceMemory<T, 1> iu_out(output.data(), size, true);

	optox::AddOperator<T, 1> op;
	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 1.0);

	op.setStream(d.stream());

	op.forward({&iu_out}, {&iu_a, &iu_b});
}

#define REGISTER_GPU(type) \
	template class AddOperatorForward<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU

template <typename T>
void AddOperatorAdjoint<T>::operator()(const GPUDevice& d, 
	typename Tensor1<T>::Tensor &output_1, 
	typename Tensor1<T>::Tensor &output_2, 
	const typename Tensor1<T>::ConstTensor &a_flat)
{
	iu::Size<1> size(a_flat.size());

	iu::LinearDeviceMemory<T, 1> iu_a(const_cast<T*>(a_flat.data()), size, true);

	iu::LinearDeviceMemory<T, 1> iu_out1(output_1.data(), size, true);
	iu::LinearDeviceMemory<T, 1> iu_out2(output_1.data(), size, true);

	optox::AddOperator<T, 1> op;
	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 1.0);

	op.setStream(d.stream());

	op.adjoint({&iu_out1, &iu_out2}, {&iu_a});
}

#define REGISTER_GPU(type) \
	template class AddOperatorAdjoint<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
