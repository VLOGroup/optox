///@file addoperator.h
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

    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual unsigned int getNumOutputsForwad()
    {
        return 1;
    }

    virtual unsigned int getNumInputsForwad()
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

    /** No copies are allowed. */
    AddOperator(AddOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(AddOperator const &) = delete;
};

} // namespace optox
