///@file act_linear.h
///@brief linear interpolation activation function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include "act.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI LinearActOperator : public IActOperator<T>
{
  public:
    /** Constructor */
    LinearActOperator(T vmin, T vmax) : IActOperator<T>(vmin, vmax)
    {
    }

    /** Destructor */
    virtual ~LinearActOperator()
    {
    }

    LinearActOperator(LinearActOperator const &) = delete;
    void operator=(LinearActOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);
};

} // namespace optox
