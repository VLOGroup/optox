///@file act.cpp
///@brief Operator for basic activation function interface
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 01.2019


#include <iu/iucore.h>
#include <iu/iumath.h>

#include "act.h"

template<typename T>
void optox::ActOperator<T, N>::computeForward(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto x = this->template getInput<T, 2>(0, inputs);
    auto w = this->template getInput<T, 2>(1, inputs);

    auto out = this->template getOutput<T, 2>(0, outputs);

    iu::math::addWeighted(*in_0, 1, *in_1, 2, *out, this->stream_);
}

template<typename T>
void optox::ActOperator<T, N>::computeAdjoint(optox::OperatorOutputVector &&outputs,
    const optox::OperatorInputVector &inputs)
{
    auto grad_out = this->template getInput<T, 2>(0, inputs);

    auto grad_x = this->template getOutput<T, 2>(0, outputs);
    auto grad_w = this->template getOutput<T, 2>(1, outputs);

    iu::math::mulC(*grad_out, 1, *grad_x, this->stream_);
    iu::math::mulC(*grad_out, 1, *grad_w, this->stream_);
}


#define REGISTER_OP(T) \
    template class optox::AddOperator<T>;

OPTOX_CALL_REAL_NUMBER_TYPES(REGISTER_OP);
#undef REGISTER_OP
