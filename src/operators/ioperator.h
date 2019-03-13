///@file ioperator.h
///@brief Interface for operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <sstream>

#include <initializer_list>
#include <vector>
#include <map>

#include "optox_api.h"
#include "utils.h"
#include "tensor/tensor.h"
#include "tensor/d_tensor.h"

#include <cuda.h>

namespace optox
{

typedef std::vector<const ITensor *> OperatorInputVector;
typedef std::vector<ITensor *> OperatorOutputVector;

/**
 * Interface for operators
 *  It defines 
 *      - the common functions that *all* operators must implement.
 *      - auxiliary helper functions
 */

class OPTOX_DLLAPI IOperator
{
  public:
    IOperator() : stream_(cudaStreamDefault)
    {
    }

    virtual ~IOperator()
    {
    }

    /** apply the operators
     * \outputs list of the operator outputs `{&out_1, ...}` which are
     *          typically of type `DTensor`
     * \inputs list of the operator inputs `{&out_1, ...}` which are
     *         typically of type `DTensor`
     */
    void forward(std::initializer_list<ITensor *> outputs,
                 std::initializer_list<const ITensor *> inputs)
    {
        if (outputs.size() != getNumOutputsForward())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the requied number!");

        if (inputs.size() != getNumInputsForward())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the requied number!");

        computeForward(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    /** apply the operator's adjoint
     * \outputs list of the operator adjoint outputs `{&out_1, ...}` which are
     *          typically of type `DTensor`
     * \inputs list of the operator adjoint inputs `{&out_1, ...}` which are
     *         typically of type `DTensor`
     */
    void adjoint(std::initializer_list<ITensor *> outputs,
                 std::initializer_list<const ITensor *> inputs)
    {
        if (outputs.size() != getNumOutputsAdjoint())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the requied number!");

        if (inputs.size() != getNumInputsAdjoint())
            THROW_OPTOXEXCEPTION("Provided number of outputs does not match the requied number!");

        computeAdjoint(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    template <typename T, unsigned int N>
    const DTensor<T, N> *getInput(unsigned int index, const OperatorInputVector &inputs)
    {
        if (index < inputs.size())
        {
            const DTensor<T, N> *t = dynamic_cast<const DTensor<T, N> *>(inputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_OPTOXEXCEPTION("Cannot cast input to desired type!");
        }
        else
            THROW_OPTOXEXCEPTION("input index out of bounds!");
    }

    template <typename T, unsigned int N>
    DTensor<T, N> *getOutput(unsigned int index, const OperatorOutputVector &outputs)
    {
        if (index < outputs.size())
        {
            DTensor<T, N> *t = dynamic_cast<DTensor<T, N> *>(outputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_OPTOXEXCEPTION("Cannot cast output to desired type!");
        }
        else
            THROW_OPTOXEXCEPTION("output index out of bounds!");
    }

    void setStream(const cudaStream_t &stream)
    {
        stream_ = stream;
    }

    cudaStream_t getStream() const
    {
        return stream_;
    }

    /** No copies are allowed. */
    IOperator(IOperator const &) = delete;

    /** No assignments are allowed. */
    void operator=(IOperator const &) = delete;

  protected:
    /** actual implementation of the forward operator
     * \outputs outputs that are computed by the forward op
     * \inputs inputs that are required to compute
     */
    virtual void computeForward(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /** actual implementation of the adjoint operator
     * \outputs outputs that are computed by the forward op
     * \inputs inputs that are required to compute
     */
    virtual void computeAdjoint(OperatorOutputVector &&outputs,
                                const OperatorInputVector &inputs) = 0;

    /** Number of rquired outputs for the forward op */
    virtual unsigned int getNumOutputsForward() = 0;
    /** Number of rquired inputs for the forward op */
    virtual unsigned int getNumInputsForward() = 0;

    /** Number of rquired outputs for the adjoint op */
    virtual unsigned int getNumOutputsAdjoint() = 0;
    /** Number of rquired inputs for the adjoint op */
    virtual unsigned int getNumInputsAdjoint() = 0;

  protected:
    cudaStream_t stream_;
};

#define OPTOX_CALL_float(m) m(float)
#define OPTOX_CALL_double(m) m(double)

#define OPTOX_CALL_REAL_NUMBER_TYPES(m) \
    OPTOX_CALL_float(m) OPTOX_CALL_double(m)

} // namespace optox
