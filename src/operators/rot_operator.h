///@file rot_operator.h
///@brief Operator rotating kernel stack
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 06.2019

#include "ioperator.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI RotOperator : public IOperator
{
  public:
    /** Constructor.
   */
    RotOperator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~RotOperator()
    {
    }

    RotOperator(RotOperator const &) = delete;
    void operator=(RotOperator const &) = delete;

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
        return 2;
    }

    virtual unsigned int getNumOutputsAdjoint()
    {
        return 1;
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return 2;
    }
};

} // namespace optox
