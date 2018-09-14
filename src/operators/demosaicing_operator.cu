///@file demosaicing_operator.cu
///@brief demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 09.07.2018


#include <iu/iucore.h>
#include <iu/iumath.h>

#include "demosaicing_operator.h"

template<typename T, optox::BayerPattern P>
__global__ void demosaicingForwardKernel(
    typename iu::LinearDeviceMemory<T, 4>::KernelData output,
    const typename iu::LinearDeviceMemory<T, 4>::KernelData input)
{
    const int x = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
    const int y = 2 * (threadIdx.y + blockIdx.y * blockDim.y);
    const int s = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x < input.size_[1] && y < input.size_[2] && s < input.size_[3])
    {
        switch (P)
        {
            case optox::BayerPattern::BGGR:
            {
                output(0, x, y, s) = input(2, x, y, s);
                output(0, x+1, y, s) = input(1, x+1, y, s);
                output(0, x, y+1, s) = input(1, x, y+1, s);
                output(0, x+1, y+1, s) = input(0, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::RGGB:
            {
                output(0, x, y, s) = input(0, x, y, s);
                output(0, x+1, y, s) = input(1, x+1, y, s);
                output(0, x, y+1, s) = input(1, x, y+1, s);
                output(0, x+1, y+1, s) = input(2, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::GBRG:
            {
                output(0, x, y, s) = input(1, x, y, s);
                output(0, x+1, y, s) = input(2, x+1, y, s);
                output(0, x, y+1, s) = input(0, x, y+1, s);
                output(0, x+1, y+1, s) = input(1, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::GRBG:
            {
                output(0, x, y, s) = input(1, x, y, s);
                output(0, x+1, y, s) = input(0, x+1, y, s);
                output(0, x, y+1, s) = input(2, x, y+1, s);
                output(0, x+1, y+1, s) = input(1, x+1, y+1, s);
                break;
            }
        }
    }
}

template<typename T>
void optox::DemosaicingOperator<T>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 4>(0, inputs);
    auto output = this->template getOutput<T, 4>(0, outputs);

    if (input->size()[0] != 3)
        THROW_IUEXCEPTION("DemosaicingOperator: input to forward must be RGB image!");

    if (output->size()[0] != 1)
        THROW_IUEXCEPTION("DemosaicingOperator: output of forward must have 1 channel!");

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid(iu::divUp(input->size()[1] / 2 + 1, dim_block.x),
                  iu::divUp(input->size()[2] / 2 + 1, dim_block.y),
                  iu::divUp(input->size()[3], dim_block.z));

    switch (this->pattern_)
    {
        case optox::BayerPattern::BGGR:
            demosaicingForwardKernel<T, optox::BayerPattern::BGGR> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::RGGB:
            demosaicingForwardKernel<T, optox::BayerPattern::RGGB> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::GBRG:
            demosaicingForwardKernel<T, optox::BayerPattern::GBRG> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::GRBG:
            demosaicingForwardKernel<T, optox::BayerPattern::GRBG> <<<dim_grid, dim_block>>>(*output, *input);
            break;
    }
    IU_CUDA_CHECK;
}

template<typename T, optox::BayerPattern P>
__global__ void demosaicingAdjointKernel(
    typename iu::LinearDeviceMemory<T, 4>::KernelData output,
    const typename iu::LinearDeviceMemory<T, 4>::KernelData input)
{
    const int x = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
    const int y = 2 * (threadIdx.y + blockIdx.y * blockDim.y);
    const int s = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (x < input.size_[1] && y < input.size_[2] && s < input.size_[3])
    {
        switch (P)
        {
            case optox::BayerPattern::BGGR:
            {
		output(2, x, y, s) = input(0, x, y, s);
                output(1, x+1, y, s) = input(0, x+1, y, s);
                output(1, x, y+1, s) = input(0, x, y+1, s);
                output(0, x+1, y+1, s) = input(0, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::RGGB:
            {
		output(0, x, y, s) = input(0, x, y, s);
                output(1, x+1, y, s) = input(0, x+1, y, s);
                output(1, x, y+1, s) = input(0, x, y+1, s);
                output(2, x+1, y+1, s) = input(0, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::GBRG:
            {	
		output(1, x, y, s) = input(0, x, y, s);
                output(2, x+1, y, s) = input(0, x+1, y, s);
                output(0, x, y+1, s) = input(0, x, y+1, s);
                output(1, x+1, y+1, s) = input(0, x+1, y+1, s);
                break;
            }
            case optox::BayerPattern::GRBG:
            {
		output(1, x, y, s) = input(0, x, y, s);
                output(0, x+1, y, s) = input(0, x+1, y, s);
                output(2, x, y+1, s) = input(0, x, y+1, s);
                output(1, x+1, y+1, s) = input(0, x+1, y+1, s);
                break;
            }
        }
    }
}

template<typename T>
void optox::DemosaicingOperator<T>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto input = this->template getInput<T, 4>(0, inputs);
    auto output = this->template getOutput<T, 4>(0, outputs);

    if (input->size()[0] != 1)
        THROW_IUEXCEPTION("DemosaicingOperator: input to adjoint must have 1 channel!");

    if (output->size()[0] != 3)
        THROW_IUEXCEPTION("DemosaicingOperator: output of adjoint must be RGB image!");

    iu::math::fill(*output, static_cast<T>(0));

    dim3 dim_block = dim3(32, 32, 1);
    dim3 dim_grid(iu::divUp(input->size()[1] / 2 + 1, dim_block.x),
                  iu::divUp(input->size()[2] / 2 + 1, dim_block.y),
                  iu::divUp(input->size()[3], dim_block.z));

    switch (this->pattern_)
    {
        case optox::BayerPattern::BGGR:
            demosaicingAdjointKernel<T, optox::BayerPattern::BGGR> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::RGGB:
            demosaicingAdjointKernel<T, optox::BayerPattern::RGGB> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::GBRG:
            demosaicingAdjointKernel<T, optox::BayerPattern::GBRG> <<<dim_grid, dim_block>>>(*output, *input);
            break;
        case optox::BayerPattern::GRBG:
            demosaicingAdjointKernel<T, optox::BayerPattern::GRBG> <<<dim_grid, dim_block>>>(*output, *input);
            break;
    }
    IU_CUDA_CHECK;
}

#define REGISTER_OP(T) \
    template class optox::DemosaicingOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
#undef REGISTER_OP_T