///@file tf_demosiaing.cu
///@brief tensorflow wrappers for the demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 04.12.2018

#include <iostream>
#include <cuda.h>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"

#include <iu/iucore/lineardevicememory.h>

#include "tf_demosaicing.h"
#include "operators/demosaicing_operator.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
void DemosaicingOperatorWrapper<T>::forward(const GPUDevice& d, 
	typename Tensor4<T>::Tensor &output, 
	const typename Tensor4<T>::ConstTensor &input)
{
	iu::Size<4> input_size({input.dimensions()[3], input.dimensions()[2], input.dimensions()[1], input.dimensions()[0]});
        iu::Size<4> output_size({output.dimensions()[3], output.dimensions()[2], output.dimensions()[1], output.dimensions()[0]});

	iu::LinearDeviceMemory<T, 4> iu_in(const_cast<T*>(input.data()), input_size, true);

	iu::LinearDeviceMemory<T, 4> iu_out(output.data(), output_size, true);

	this->op.setStream(d.stream());

	this->op.forward({&iu_out}, {&iu_in});
}

template <typename T>
void DemosaicingOperatorWrapper<T>::adjoint(const GPUDevice& d, 
	typename Tensor4<T>::Tensor &output, 
	const typename Tensor4<T>::ConstTensor &input)
{
        iu::Size<4> input_size({input.dimensions()[3], input.dimensions()[2], input.dimensions()[1], input.dimensions()[0]});
        iu::Size<4> output_size({output.dimensions()[3], output.dimensions()[2], output.dimensions()[1], output.dimensions()[0]});;

	iu::LinearDeviceMemory<T, 4> iu_in(const_cast<T*>(input.data()), input_size, true);

	iu::LinearDeviceMemory<T, 4> iu_out(output.data(), output_size, true);

	this->op.setStream(d.stream());

	this->op.adjoint({&iu_out}, {&iu_in});
}

#define REGISTER_GPU(type) \
	template class DemosaicingOperatorWrapper<type>;

REGISTER_GPU(float);
REGISTER_GPU(double);

#undef REGISTER_GPU
