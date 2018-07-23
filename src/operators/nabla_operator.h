///@file nabla_operator.h
///@brief Operator that computes the forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

namespace optox
{

template <typename T, unsigned int N>
class OPTOX_DLLAPI NablaOperator : public IOperator
{
  public:
    /** Constructor.
   */
    NablaOperator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~NablaOperator()
    {
    }

    NablaOperator(NablaOperator const &) = delete;
    void operator=(NablaOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual unsigned int getNumOutputsForward()
    {
        return 1;
    }

    virtual unsigned int getNumInputsForward()
    {
        return 1;
    }

    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return 1;
    }
};

} // namespace optox
