///@file pad2d_operator.h
///@brief Operator that pads an image given with symmetric boundary conndition
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.202

#include "ioperator.h"

namespace optox
{

template <typename T>
class OPTOX_DLLAPI Pad2dOperator : public IOperator
{
  public:
    /** Constructor.
   */
    Pad2dOperator(int left, int right, int top, int bottom) : IOperator(),
        left_(left), right_(right), top_(top), bottom_(bottom)
    {
        std::cout << "creating pad op" << std::endl;
    }

    /** Destructor */
    virtual ~Pad2dOperator()
    {
    }

    Pad2dOperator(Pad2dOperator const &) = delete;
    void operator=(Pad2dOperator const &) = delete;

    int paddingX() const
    {
        return this->left_ + this->right_;
    }

    int paddingY() const
    {
        return this->top_ + this->bottom_;
    }

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
    
  private:
    int left_;
    int right_;
    int top_;
    int bottom_;
};

} // namespace optox
