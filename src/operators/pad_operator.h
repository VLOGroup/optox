///@file pad2d_operator.h
///@brief Operator that pads an image given with symmetric boundary conndition
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.202

#include "ioperator.h"

namespace optox
{

enum PaddingMode {symmetric, reflect, replicate};

template <typename T>
class OPTOX_DLLAPI Pad2dOperator : public IOperator
{
  public:
    /** Constructor.
    */
    Pad2dOperator(int left, int right, int top, int bottom, const std::string &mode) : IOperator(),
        left_(left), right_(right), top_(top), bottom_(bottom)
    {
        if (mode == "symmetric") mode_ = PaddingMode::symmetric;
        else if (mode == "reflect") mode_ = PaddingMode::reflect;
        else if (mode == "replicate") mode_ = PaddingMode::replicate;
        else THROW_OPTOXEXCEPTION("Pad2dOperator: invalid mode!");
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
    PaddingMode mode_;
};


template <typename T>
class OPTOX_DLLAPI Pad3dOperator : public IOperator
{
  public:
    /** Constructor.
    */
    Pad3dOperator(int left, int right, int top, int bottom, int front, int back, const std::string &mode) : IOperator(),
        left_(left), right_(right), top_(top), bottom_(bottom), front_(front), back_(back)
    {
        if (mode == "symmetric") mode_ = PaddingMode::symmetric;
        else if (mode == "reflect") mode_ = PaddingMode::reflect;
        else if (mode == "replicate") mode_ = PaddingMode::replicate;
        else THROW_OPTOXEXCEPTION("Pad3dOperator: invalid mode!");
    }

    /** Destructor */
    virtual ~Pad3dOperator()
    {
    }

    Pad3dOperator(Pad3dOperator const &) = delete;
    void operator=(Pad3dOperator const &) = delete;

    int paddingX() const
    {
        return this->left_ + this->right_;
    }

    int paddingY() const
    {
        return this->top_ + this->bottom_;
    }

    int paddingZ() const
    {
        return this->front_ + this->back_;
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
    int front_;
    int back_;
    PaddingMode mode_;
};


} // namespace optox
