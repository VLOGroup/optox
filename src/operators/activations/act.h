///@file act.h
///@brief Operator for basic activation function interface
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#pragma once

#include "../ioperator.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI IActOperator : public IOperator
{
  public:
    /** Constructor */
    IActOperator(T vmin, T vmax) : IOperator(), vmin_(vmin), vmax_(vmax)
    {
    }

    /** Destructor */
    virtual ~IActOperator()
    {
    }

    IActOperator(IActOperator const &) = delete;
    void operator=(IActOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    void checkSize(const iu::Size<2> input_size, const iu::Size<2> weights_size)
    {
        if (input_size[1] != weights_size[1])
            throw std::runtime_error("Activation operator: input and weights size do not match!");
    }

    virtual unsigned int getNumOutputsForward()
    {
        return 1;
    }

    virtual unsigned int getNumInputsForward()
    {
        return 2;
    }

    virtual unsigned int getNumOutputsAdjoint()
    {
        return 2;
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return 3;
    }

  protected:
    T vmin_;
    T vmax_;
};

} // namespace optox
