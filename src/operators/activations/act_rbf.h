///@file act_rbf.h
///@brief Gaussian radial basis function operator
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019

#include "act.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI RBFActOperator : public IActOperator<T>
{
  public:
    /** Constructor */
    RBFActOperator(T vmin, T vmax) : IActOperator<T>(vmin, vmax)
    {
    }

    /** Destructor */
    virtual ~RBFActOperator()
    {
    }

    RBFActOperator(RBFActOperator const &) = delete;
    void operator=(RBFActOperator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);
};


template <typename T>
class OPTOX_DLLAPI RBFAct2Operator : public IAct2Operator<T>
{
  public:
    /** Constructor */
    RBFAct2Operator(T vmin, T vmax) : IAct2Operator<T>(vmin, vmax)
    {
    }

    /** Destructor */
    virtual ~RBFAct2Operator()
    {
    }

    RBFAct2Operator(RBFAct2Operator const &) = delete;
    void operator=(RBFAct2Operator const &) = delete;

  protected:
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);

    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs);
};

} // namespace optox
