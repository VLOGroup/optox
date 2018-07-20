///@file add_operator.h
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

namespace optox
{

template <typename T, unsigned int N>
class OPTOX_DLLAPI AddOperator : public IOperator
{
  public:
    /** Constructor.
   */
    AddOperator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~AddOperator()
    {
    }

    AddOperator(AddOperator const &) = delete;
    void operator=(AddOperator const &) = delete;

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
        return 2;
    }

    virtual unsigned int getNumInputsAdjoint()
    {
        return 1;
    }
};

} // namespace optox
