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
void applyAddOperator<T>::operator()(const GPUDevice& d, 
	typename Tensor1<T>::Tensor &output, 
	const typename Tensor1<T>::ConstTensor &a_flat, 
	const typename Tensor1<T>::ConstTensor &b_flat)
{
	iu::Size<1> size(output.size());

	// FIXME: support const LinearDeviceMemory
	iu::LinearDeviceMemory<T, 1> iu_a(const_cast<T*>(a_flat.data()), size, true);
	iu::LinearDeviceMemory<T, 1> iu_b(const_cast<T*>(b_flat.data()), size, true);

	iu::LinearDeviceMemory<T, 1> iu_out(output.data(), size, true);

	optox::AddOperator<T, 1> op;

	op.setParameter("w_1", 1.0);
	op.setParameter("w_2", 2);

	op.appendInput(iu_a);
	op.appendInput(iu_b);

	op.appendOutput(iu_out);

	op.setStream(d.stream());

	op.apply();
}

#define REGISTER_GPU(type) \
	template class applyAddOperator<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU