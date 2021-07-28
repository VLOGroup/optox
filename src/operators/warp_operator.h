///@file warp_operator.h
///@brief Operator that warps an image given a flow field
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.2019

#include "ioperator.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI WarpOperator : public IOperator
{
  public:
    /** Constructor.
   */
    WarpOperator() : IOperator()
    {
    }

    /** Destructor */
    virtual ~WarpOperator()
    {
    }

    WarpOperator(WarpOperator const &) = delete;
    void operator=(WarpOperator const &) = delete;

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
