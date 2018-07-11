///@file addoperator.cpp
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018


#include <iu/iucore.h>

#include "add_operator.h"
#include "optox_math.h"

template<typename T, unsigned int N>
void optox::AddOperator<T, N>::apply()
{
    auto in_0 = this->template getInput<T, N>(0);
    auto in_1 = this->template getInput<T, N>(1);

    auto out = this->template getOutput<T, N>(0);

    T w_1 = this->config_.template getValue<T>("w_1");
    T w_2 = this->config_.template getValue<T>("w_2");

    optox::math::addWeighted<T, N>(*in_0, w_1, *in_1, w_2, *out, this->stream_);
}

template<typename T, unsigned int N>
void optox::AddOperatorAdjoint<T, N>::apply()
{
    auto in_0 = this->template getInput<T, N>(0);

    auto out_0 = this->template getOutput<T, N>(0);
    auto out_1 = this->template getOutput<T, N>(0);

    T w_1 = this->config_.template getValue<T>("w_1");
    T w_2 = this->config_.template getValue<T>("w_2");

    optox::math::mulC<T, N>(*in_0, w_1, *out_0, this->stream_);
    optox::math::mulC<T, N>(*in_0, w_2, *out_1, this->stream_);
}


#define REGISTER_OP_T(T, N) \
    template class optox::AddOperator<T, N>; \
    template class optox::AddOperatorAdjoint<T, N>;

#define REGISTER_OP(T) \
    REGISTER_OP_T(T, 1) \
    REGISTER_OP_T(T, 2) \
    REGISTER_OP_T(T, 3) \
    REGISTER_OP_T(T, 4)

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
#undef REGISTER_OP_T
