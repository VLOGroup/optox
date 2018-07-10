///@file addoperator.h
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include "ioperator.h"

namespace optox{

template<typename T, unsigned int N>
class OPTOX_DLLAPI IAddOperator: public IOperator<T, N, T, N>
{
public:
  /** Constructor.
   */
  IAddOperator() : IOperator<T, N, T, N>()
  {
  }

  /** Destructor */
  virtual ~IAddOperator()
  {
  }

  /** Apply the operation on src and store it in dst. Additionally make checks
   * on operator.
   */
  virtual void apply() = 0;

  /** No copies are allowed. */
  IAddOperator(IAddOperator const&) = delete;

  /** No assignments are allowed. */
  void operator=(IAddOperator const&) = delete;
};


template<typename T, unsigned int N>
class OPTOX_DLLAPI AddOperator: public IAddOperator<T, N>
{
public:
  /** Constructor.
   */
  AddOperator() : IAddOperator<T, N>()
  {
  }

  /** Destructor */
  virtual ~AddOperator()
  {
  }

  /** Apply the operation on src and store it in dst. Additionally make checks
   * on operator.
   */
  virtual void apply();

  /** No copies are allowed. */
  AddOperator(AddOperator const&) = delete;

  /** No assignments are allowed. */
  void operator=(AddOperator const&) = delete;
};


template<typename T, unsigned int N>
class OPTOX_DLLAPI AddOperatorAdjoint: public IAddOperator<T, N>
{
public:
  /** Constructor.
   */
  AddOperatorAdjoint() : IAddOperator<T, N>()
  {
  }

  /** Destructor */
  virtual ~AddOperatorAdjoint()
  {
  }

  /** Apply the operation on src and store it in dst. Additionally make checks
   * on operator.
   */
  virtual void apply();

  /** No copies are allowed. */
  AddOperatorAdjoint(AddOperatorAdjoint const&) = delete;

  /** No assignments are allowed. */
  void operator=(AddOperatorAdjoint const&) = delete;
};

}
