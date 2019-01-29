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

#include <iu/iucore.h>

#include "optox_api.h"

#include <cuda.h>

namespace optox
{

typedef std::vector<const iu::ILinearMemory *> OperatorInputVector;
typedef std::vector<iu::ILinearMemory *> OperatorOutputVector;

typedef std::map<std::string, std::string> OperatorConfigDict;

/** OpConfig holds the configuration parameters for the Operator.
 */
class OPTOX_DLLAPI OperatorConfig
{
  public:
    OperatorConfig(const OperatorConfigDict &config = OperatorConfigDict()) : dict_(config)
    {
    }

    /** Get the value for a specific configuration parameter.
     * \param key configuration parameter
     * \return configuration value
     */
    template <typename T>
    T getValue(const std::string &key) const
    {
        auto iter = dict_.find(key);
        if (iter == dict_.end())
            THROW_IUEXCEPTION("key not found!");
        else
        {
            std::stringstream ss;
            ss << iter->second;
            T ret;
            ss >> ret;
            if (ss.fail())
                THROW_IUEXCEPTION("could not parse!");
            return ret;
        }
    }

    /** Set the value for a specifig configuration parameter
     * \param key configartion parameter
     * \val value for the specified parameter
     */
    template <typename T>
    void setValue(const std::string &key, T val)
    {
        std::stringstream ss;
        ss << val;
        dict_[key] = ss.str();
    }

    /** Check if the dictionary has a specific configuration parameter.
     * \param key configuration parameter
     * \return true if key is in the dictionary.
     */
    bool hasKey(const std::string &key) const
    {
        auto iter = dict_.find(key);
        if (iter == dict_.end())
            return false;
        else
            return true;
    }

    int size() const
    {
        return dict_.size();
    }

    friend std::ostream &operator<<(std::ostream &out, OperatorConfig const &conf)
    {
        unsigned int i = 0;
        for (auto iter = conf.dict_.begin(); iter != conf.dict_.end(); ++iter)
            out << iter->first << ":" << iter->second
                << ((++i < conf.dict_.size()) ? "," : "");
        return out;
    }

  private:
    OperatorConfigDict dict_;
};

/**
 * Interface for operators
 *  It defines 
 *      - the common functions that *all* operators must implement.
 *      - auxiliary helper functions
 */

class OPTOX_DLLAPI IOperator
{
  public:
    IOperator()
        : config_(), stream_(cudaStreamDefault)
    {
    }

    virtual ~IOperator()
    {
    }

    /** apply the operators
     * \outputs list of the operator outputs `{&out_1, ...}` which are
     *          typically of type `iu::LinearDeviceMemory`
     * \inputs list of the operator inputs `{&out_1, ...}` which are
     *         typically of type `iu::LinearDeviceMemory`
     */
    void forward(std::initializer_list<iu::ILinearMemory *> outputs,
                 std::initializer_list<const iu::ILinearMemory *> inputs)
    {
        if (outputs.size() != getNumOutputsForward())
            THROW_IUEXCEPTION("Provided number of outputs does not match the requied number!");

        if (inputs.size() != getNumInputsForward())
            THROW_IUEXCEPTION("Provided number of outputs does not match the requied number!");

        computeForward(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    /** apply the operator's adjoint
     * \outputs list of the operator adjoint outputs `{&out_1, ...}` which are
     *          typically of type `iu::LinearDeviceMemory`
     * \inputs list of the operator adjoint inputs `{&out_1, ...}` which are
     *         typically of type `iu::LinearDeviceMemory`
     */
    void adjoint(std::initializer_list<iu::ILinearMemory *> outputs,
                 std::initializer_list<const iu::ILinearMemory *> inputs)
    {
        if (outputs.size() != getNumOutputsAdjoint())
            THROW_IUEXCEPTION("Provided number of outputs does not match the requied number!");

        if (inputs.size() != getNumInputsAdjoint())
            THROW_IUEXCEPTION("Provided number of outputs does not match the requied number!");

        computeAdjoint(OperatorOutputVector(outputs), OperatorInputVector(inputs));
    }

    template <typename T, unsigned int N>
    const iu::LinearDeviceMemory<T, N> *getInput(int index, const OperatorInputVector &inputs)
    {
        if (index >= 0 && index < inputs.size())
        {
            const iu::LinearDeviceMemory<T, N> *t = dynamic_cast<const iu::LinearDeviceMemory<T, N> *>(inputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_IUEXCEPTION("Cannot cast input to desired type!");
        }
        else
            THROW_IUEXCEPTION("input index out of bounds!");
    }

    template <typename T, unsigned int N>
    iu::LinearDeviceMemory<T, N> *getOutput(int index, const OperatorOutputVector &outputs)
    {
        if (index >= 0 && index < outputs.size())
        {
            iu::LinearDeviceMemory<T, N> *t = dynamic_cast<iu::LinearDeviceMemory<T, N> *>(outputs[index]);
            if (t != nullptr)
                return t;
            else
                THROW_IUEXCEPTION("Cannot cast output to desired type!");
        }
        else
            THROW_IUEXCEPTION("output index out of bounds!");
    }

    void setConfig(const OperatorConfigDict &dict)
    {
        config_ = OperatorConfig(dict);
    }

    template <typename T>
    T getParameter(const char *name) const
    {
        return config_.getValue<T>(std::string(name));
    }

    template <typename T>
    void setParameter(const char *name, const T &val)
    {
        config_.setValue<T>(std::string(name), val);
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
    OperatorConfig config_;

    cudaStream_t stream_;
};

#define OPTOX_CALL_float(m) m(float)
#define OPTOX_CALL_double(m) m(double)

#define OPTOX_CALL_REAL_NUMBER_TYPES(m) \
    OPTOX_CALL_float(m) OPTOX_CALL_double(m)

} // namespace optox
