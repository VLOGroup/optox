///@file demosaicing_operator.h
///@brief demosaicing operator
///@author Joana Grah <joana.grah@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

#pragma once

namespace optox
{

enum BayerPattern
{
    RGGB = 0,
    BGGR,
    GBRG,
    GRBG,
};

BayerPattern fromString(const std::string & str)
{
    if (str == "RGGB")
        return BayerPattern::RGGB;
    else if (str == "BGGR")
        return BayerPattern::BGGR;
    else if (str == "GBRG")
        return BayerPattern::GBRG;
    else if (str == "GRBG")
        return BayerPattern::GRBG;
    else 
        throw std::runtime_error("Invalid bayer pattern!");
}

template <typename T>
class OPTOX_DLLAPI DemosaicingOperator : public IOperator
{
  public:
    /** Constructor.
   */
    DemosaicingOperator(const std::string &pattern) : IOperator(), 
        pattern_(fromString(pattern))
    {
    }

    /** Destructor */
    virtual ~DemosaicingOperator()
    {
    }

    DemosaicingOperator(DemosaicingOperator const &) = delete;
    void operator=(DemosaicingOperator const &) = delete;

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
    BayerPattern pattern_;
};

} // namespace optox
