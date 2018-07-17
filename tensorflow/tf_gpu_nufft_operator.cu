#include <iostream>
#include <cuda.h>

#define EIGEN_USE_GPU
#include "tensorflow/core/framework/tensor.h"

#include <iu/iucore/lineardevicememory.h>

#include "tf_gpu_nufft_operator.h"

#include <gpuNUFFT_operator_factory.hpp>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

void applyGpuNufftForwardOperator::operator()(const GPUDevice& d, 
	typename tensorflow::TTypes<tensorflow::complex64,3>::Tensor &rawdata, 
	const typename tensorflow::TTypes<tensorflow::complex64,3>::ConstTensor &img, 
	const typename tensorflow::TTypes<tensorflow::complex64,4>::ConstTensor &sensitivities,
    const typename tensorflow::TTypes<float,3>::ConstTensor &trajectory,
    const typename tensorflow::TTypes<float,2>::ConstTensor &dcf,
    const int& osf,
    const int& sector_width,
    const int& kernel_width,
    const int& img_dim)
{
		// create NUFFT operator
		gpuNUFFT::Dimensions img_dims;
		img_dims.width = img_dim;
		img_dims.height = img_dim;
		img_dims.depth = 0;

		gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,d.stream());

		// loop over samples
		int samples = dcf.dimensions()[0];
		//std::cout << "samples: " << samples << std::endl;

		gpuNUFFT::Array<float> trajectory_gpunufft;
		// std::cout << "dcf length: " << dcf.dimensions()[1] << std::endl;
		trajectory_gpunufft.dim.length = dcf.dimensions()[1];
		int trajectory_offset = trajectory.size() / samples;

		gpuNUFFT::Array<float> dcf_gpunufft;
		dcf_gpunufft.dim.length = dcf.dimensions()[1];
		int dcf_offset = dcf.size() / samples;

		gpuNUFFT::Array<float2> sensitivities_gpunufft;
		sensitivities_gpunufft.dim = img_dims;
		// std::cout << "nCoils: " << sensitivities.dimensions()[1] << std::endl;
		sensitivities_gpunufft.dim.channels = sensitivities.dimensions()[1];
		int sensitivities_offset = sensitivities.size() / samples;

		gpuNUFFT::Array<float2> img_gpunufft;
		img_gpunufft.dim = img_dims;
		int img_offset = img.size() / samples;

		gpuNUFFT::Array<float2> rawdata_gpunufft;
		// std::cout << "rawdata length: " << rawdata.dimensions()[2] << std::endl;
		// std::cout << "rawdata channels: " << rawdata.dimensions()[1] << std::endl;
		rawdata_gpunufft.dim.length = rawdata.dimensions()[2];
		rawdata_gpunufft.dim.channels = rawdata.dimensions()[1];
		int rawdata_offset = rawdata.size() / samples;

		// std::cout << "trajectory_offset " << trajectory_offset << std::endl;
		// std::cout << "dcf_offset " << dcf_offset << std::endl;
		// std::cout << "sensitivities_offset " << sensitivities_offset << std::endl;
		// std::cout << "rawdata_offset " << rawdata_offset << std::endl;
		// std::cout << "image_offset " << img_offset << std::endl;

		for (int n = 0; n < samples; n++)
		{
			trajectory_gpunufft.data = const_cast<float*>(reinterpret_cast<const float*>(trajectory.data() + n * trajectory_offset));
			dcf_gpunufft.data = const_cast<float*>(reinterpret_cast<const float*>(dcf.data() + n * dcf_offset));
			sensitivities_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(sensitivities.data())) + n * sensitivities_offset;

			gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(trajectory_gpunufft,
													dcf_gpunufft, 
													sensitivities_gpunufft,
													kernel_width,
													sector_width,
													osf,
													img_dims);
			
			img_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(img.data())) + n * img_offset;
			rawdata_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(rawdata.data())) + n * rawdata_offset;

			nufft_op->performForwardGpuNUFFT(img_gpunufft, rawdata_gpunufft);

			delete nufft_op;
		}
}

void applyGpuNufftAdjointOperator::operator()(const GPUDevice& d, 
	typename tensorflow::TTypes<tensorflow::complex64,3>::Tensor &img, 
	const typename tensorflow::TTypes<tensorflow::complex64,3>::ConstTensor &rawdata, 
	const typename tensorflow::TTypes<tensorflow::complex64,4>::ConstTensor &sensitivities,
    const typename tensorflow::TTypes<float,3>::ConstTensor &trajectory,
    const typename tensorflow::TTypes<float,2>::ConstTensor &dcf,
    const int& osf,
    const int& sector_width,
    const int& kernel_width,
    const int& img_dim)
{
		// create NUFFT operator
		gpuNUFFT::Dimensions img_dims;
		img_dims.width = img_dim;
		img_dims.height = img_dim;
		img_dims.depth = 0;

		gpuNUFFT::GpuNUFFTOperatorFactory factory(true,true,true,false,d.stream());
		
		// loop over samples
		int samples = dcf.dimensions()[0];
		//std::cout << "samples: " << samples << std::endl;

		gpuNUFFT::Array<float> trajectory_gpunufft;
		//std::cout << "dcf length: " << dcf.dimensions()[1] << std::endl;
		trajectory_gpunufft.dim.length = dcf.dimensions()[1];
		int trajectory_offset = trajectory.size() / samples;

		gpuNUFFT::Array<float> dcf_gpunufft;
		dcf_gpunufft.dim.length = dcf.dimensions()[1];
		int dcf_offset = dcf.size() / samples;

		gpuNUFFT::Array<float2> sensitivities_gpunufft;
		sensitivities_gpunufft.dim = img_dims;
		//std::cout << "nCoils: " << sensitivities.dimensions()[1] << std::endl;
		sensitivities_gpunufft.dim.channels = sensitivities.dimensions()[1];
		int sensitivities_offset = sensitivities.size() / samples;

		gpuNUFFT::Array<float2> img_gpunufft;
		img_gpunufft.dim = img_dims;
		int img_offset = img.size() / samples;

		gpuNUFFT::Array<float2> rawdata_gpunufft;
		// std::cout << "rawdata length: " << rawdata.dimensions()[2] << std::endl;
		// std::cout << "rawdata channels: " << rawdata.dimensions()[1] << std::endl;
		rawdata_gpunufft.dim.length = rawdata.dimensions()[2];
		rawdata_gpunufft.dim.channels = rawdata.dimensions()[1];
		int rawdata_offset = rawdata.size() / samples;

		// std::cout << "trajectory_offset " << trajectory_offset << std::endl;
		// std::cout << "dcf_offset " << dcf_offset << std::endl;
		// std::cout << "sensitivities_offset " << sensitivities_offset << std::endl;
		// std::cout << "rawdata_offset " << rawdata_offset << std::endl;
		// std::cout << "img_offset " << img_offset << std::endl;

		for (int n = 0; n < samples; n++)
		{
			trajectory_gpunufft.data = const_cast<float*>(reinterpret_cast<const float*>(trajectory.data() + n * trajectory_offset));
			dcf_gpunufft.data = const_cast<float*>(reinterpret_cast<const float*>(dcf.data() + n * dcf_offset));
			sensitivities_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(sensitivities.data())) + n * sensitivities_offset;

			gpuNUFFT::GpuNUFFTOperator* nufft_op = factory.createGpuNUFFTOperator(trajectory_gpunufft,
													dcf_gpunufft, 
													sensitivities_gpunufft,
													kernel_width,
													sector_width,
													osf,
													img_dims);
			
			img_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(img.data())) + n * img_offset;
			rawdata_gpunufft.data = const_cast<float2*>(reinterpret_cast<const float2*>(rawdata.data())) + n * rawdata_offset;

			nufft_op->performGpuNUFFTAdj(rawdata_gpunufft, img_gpunufft);

			delete nufft_op;
		}
}