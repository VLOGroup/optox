///@file ioperator.h
///@brief Interface for operators
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <map>

#include <iu/iumath/typetraits.h>
#include <iu/iucore.h>

#include "optox_api.h"

#include <cuda.h>


namespace optox{

#define THROW_IUEXCEPTION(str) throw IuException(str, __FILE__, __FUNCTION__, __LINE__)

class OPTOX_DLLAPI ITensor
{
public:
  ITensor()
  {
  }

  virtual ~ITensor()
  {
  }

};


typedef std::vector<ITensor *> OperatorIOs;

template<typename T, unsigned int N>
class OPTOX_DLLAPI Tensor : public ITensor
{
public:
  Tensor(iu::LinearHostMemory<T, N> &host_mem): device_mem_(nullptr)
  {
    device_mem_ = new iu::LinearDeviceMemory<T, N>(host_mem.size());
    iu::copy(&host_mem, device_mem_);
  }

  Tensor(iu::LinearDeviceMemory<T, N> &dev_mem, bool ext_data_pointer = false): device_mem_(nullptr)
  {
    device_mem_ = new iu::LinearDeviceMemory<T, N>(dev_mem.data(), dev_mem.size(), ext_data_pointer);
  }

  Tensor(T *dev_ptr, const iu::Size<N> &size, bool ext_data_pointer = false): device_mem_(nullptr)
  {
    device_mem_ = new iu::LinearDeviceMemory<T, N>(dev_ptr, size, ext_data_pointer);
  }

  virtual ~Tensor()
  {
    if (device_mem_ != nullptr)
      delete device_mem_;
  }

  iu::LinearDeviceMemory<T, N> *getLinearDeviceMemory() const
  {
    return device_mem_;
  }

private:
  iu::LinearDeviceMemory<T, N> *device_mem_;
};


typedef std::map<std::string, std::string> OperatorConfigDict;

/** OpConfig holds the configuration parameters for the Operator.
 */
class OPTOX_DLLAPI OperatorConfig
{
public:
  OperatorConfig(const OperatorConfigDict &config = OperatorConfigDict()): dict_(config)
  {
    for (const auto& kv : dict_)
      std::cout << kv.first << ": " << kv.second << std::endl;
  }

  /** Get the value for a specific configuration parameter.
   * \param key configuration parameter
   * \return configuration value
   */
  template<typename T>
  T getValue(const std::string &key) const
  {
    for (const auto& kv : dict_)
      std::cout << kv.first << ": " << kv.second << std::endl;
    auto iter = dict_.find(key);
    if (iter == dict_.end())
    {
      std::cout << "key=" << key << std::endl;
      THROW_IUEXCEPTION("key not found!");
    }
    else
    {
      std::stringstream ss;
      ss << iter->second;
      T ret;
      ss >> ret;
      std::cout << "key=" << key << " val=" << iter->second << " ret=" << ret << std::endl;
      if (ss.fail())
        THROW_IUEXCEPTION("could not parse!");
      return ret;
    }
  }

  template<typename T>
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

  /** Get size of the dictionary
   */
  int size() const
  {
    return dict_.size();
  }

  /** Overload operator<< for pretty printing.
   */
  friend std::ostream& operator<<(std::ostream & out, OperatorConfig const& conf)
  {
    int i = 0;
    for (auto iter = conf.dict_.begin(); iter != conf.dict_.end(); ++iter)
      out << iter->first << ":" << iter->second
          << ((++i < conf.dict_.size()) ? "," : "");
    return out;
  }

private:
  /** Dictionary */
  OperatorConfigDict dict_;
};


template<typename Tin, unsigned int Nin, typename Tout, unsigned int Nout>
class OPTOX_DLLAPI IOperator
{
public:
  /** Constructor.
   */
  IOperator(): config_(), inputs_(), outputs_(), stream_(cudaStreamDefault)
  {
  }

  /** Destructor */
  virtual ~IOperator()
  {
    for (auto i: inputs_)
      if (i != nullptr)
        delete i;

    for (auto i: outputs_)
      if (i != nullptr)
        delete i;
  }

  /** Apply the operation
   */
  virtual void apply() = 0;

  template<typename T, unsigned int N>
  void appendInput(iu::LinearDeviceMemory<T, N> &input, bool copy = true)
  {
    inputs_.push_back(new Tensor<T, N>(input, not copy));
  }

  template<typename T, unsigned int N>
  iu::LinearDeviceMemory<T, N> *getInput(int index)
  {
    if (index >= 0 && index < inputs_.size())
    {
      Tensor<T, N> *t = dynamic_cast<Tensor<T, N> *>(inputs_[index]);
      if (t != nullptr)
        return t->getLinearDeviceMemory();
      else
        THROW_IUEXCEPTION("Cannot cast input to desired type!");
    }
    else
      THROW_IUEXCEPTION("Inputss index out of bounds!");
  }

  template<typename T, unsigned int N>
  void appendOutput(iu::LinearDeviceMemory<T, N> &output, bool copy = true)
  {
    outputs_.push_back(new Tensor<T, N>(output, not copy));
  }

  template<typename T, unsigned int N>
  iu::LinearDeviceMemory<T, N> *getOutput(int index)
  {
    if (index >= 0 && index < inputs_.size())
    {
      Tensor<T, N> *t = dynamic_cast<Tensor<T, N> *>(outputs_[index]);
      if (t != nullptr)
        return t->getLinearDeviceMemory();
      else
        THROW_IUEXCEPTION("Cannot cast output to desired type!");
    }
    else
      THROW_IUEXCEPTION("Outputss index out of bounds!");
  }

  void setConfig(const OperatorConfigDict &dict)
  {
    config_ = OperatorConfig(dict);
  }

  template<typename T>
  T getParameter(const char *name) const
  {
    return config_.getValue<T>(std::string(name));
  }

  template<typename T>
  void setParameter(const char *name, T val)
  {
    config_.setValue<T>(std::string(name), val);
  }

  void setStream(const cudaStream_t &stream)
  {
    stream_ = stream;
  }

  const cudaStream_t getStream() const
  {
    return stream_;
  }

  /** No copies are allowed. */
  IOperator(IOperator const&) = delete;

  /** No assignments are allowed. */
  void operator=(IOperator const&) = delete;

protected:
  /** Operator configuration */
  OperatorConfig config_;

  /** Operator inputs */
  OperatorIOs inputs_;

  /** Operator outputs */
  OperatorIOs outputs_;

  cudaStream_t stream_;
};

#define OPTOX_CALL_float(m) m(float)
#define OPTOX_CALL_double(m) m(double)

#define OPTOX_CALL_REAL_NUMBER_TYPES(m) \
   OPTOX_CALL_float(m) OPTOX_CALL_double(m)

}

/*
namespace utils{

template<typename T unsigned int Nin, unsigned int Nout>
typename std::enable_if<!
    std::is_same<typename Tin, float2>::value && !std::is_same<Tin, double2>::value, ResultType>::type 
    testAdjointness(const iu::LinearDeviceMemory<T, Nin> &u, 
                    const iu::LinearDeviceMemory<T, Nout> &Au, 
                    const iu::LinearDeviceMemory<T, Nout> &p, 
                    const iu::LinearDeviceMemory<T, Nin> &Atp)
{
  T lhs, rhs;
  iu::math::dotProduct(Au, p, lhs);
  iu::math::dotProduct(u, Atp, rhs);

  std::cout << "<Au,p>=" << lhs << std::endl;
  std::cout << "<u,Atp>=" << rhs << std::endl;
  std::cout << "diff= " << abs(lhs - rhs) << std::endl;
  if(abs(lhs - rhs) < 1e-12)
    std::cout << "TEST PASSED" << std::endl;
  else
    std::cout << "TEST FAILED" << std::endl;
  std::cout << std::endl;
}

template<typename T unsigned int Nin, unsigned int Nout>
typename std::enable_if<
    std::is_same<typename Tin, float2>::value || std::is_same<Tin, double2>::value, ResultType>::type 
    testAdjointness(const iu::LinearDeviceMemory<T, Nin> &u, 
                    const iu::LinearDeviceMemory<T, Nout> &Au, 
                    const iu::LinearDeviceMemory<T, Nout> &p, 
                    const iu::LinearDeviceMemory<T, Nin> &Atp)
{
  T lhs, rhs;
  iu::math::complex::dotProduct(Au, p, lhs);
  iu::math::complex::dotProduct(u, Atp, rhs);

  std::cout << "<Au,p>=" << lhs << std::endl;
  std::cout << "<u,Atp>=" << rhs << std::endl;
  std::cout << "diff: x= " << abs(lhs.x - rhs.x) << " y=" << abs(lhs.y - rhs.y) << std::endl;

  if(abs(lhs.x - rhs.x) < 1e-12 && abs(lhs.y - rhs.y) < 1e-12)
  {
    std::cout << "TEST PASSED" << std::endl;
  }
  else
  {
    std::cout << "TEST FAILED" << std::endl;
  }
  std::cout << std::endl;
}
}

#define TEST_ADJOINTNESS(u, Au, p, Atp) utils::testAdjointness(u, Au, p, Atp)
*/

