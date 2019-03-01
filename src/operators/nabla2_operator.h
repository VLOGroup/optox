///@file nabla2_operator.h
///@brief Operator that computes the second order forward differences along all dimensions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 02.2019

#include "ioperator.h"

namespace optox
{

template <typename T, unsigned int N>
class OPTOX_DLLAPI Nabla2Operator : public IOperator
{
  public:
    /** Constructor.
   */
    Nabla2Operator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~Nabla2Operator()
    {
    }

    Nabla2Operator(Nabla2Operator const &) = delete;
    void operator=(Nabla2Operator const &) = delete;

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
