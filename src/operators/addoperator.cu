///@file addoperator.cpp
///@brief Operator that adds two inputs and returns the result
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#include "addoperator.h"

#include <iu/iucore.h>
#include <iu/iumath.h>

// #include <thrust/fill.h>
#include "optoxmath.h"

template<typename T, unsigned int N>
void optox::AddOperator<T, N>::apply()
{
    std::cout << "addop apply" << std::endl;
    auto in_0 = this->template getInput<T, N>(0);
    auto in_1 = this->template getInput<T, N>(1);
    std::cout << "addop got inputs" << std::endl;

    auto out = this->template getOutput<T, N>(0);
    std::cout << "addop got outputs" << std::endl;

    T w_1 = this->config_.template getValue<T>("w_1");
    T w_2 = this->config_.template getValue<T>("w_2");
    std::cout << "addop weights" << std::endl;

    // iu::math::addWeighted(*in_0, w_1, *in_1, w_2, *out);
    std::cout << "optox: " << out->data() << std::endl;
    // iu::math::fill(*out, 1);
    optox::math::fill(*out, 1, this->stream_);
    std::cout << "addop done" << std::endl;
}

template<typename T, unsigned int N>
void optox::AddOperatorAdjoint<T, N>::apply()
{
    auto in_0 = this->template getInput<T, N>(0);

    auto out_0 = this->template getOutput<T, N>(0);
    auto out_1 = this->template getOutput<T, N>(0);

    T w_1 = this->config_.template getValue<T>("w_1");
    T w_2 = this->config_.template getValue<T>("w_2");

    iu::math::mulC(*in_0, w_1, *out_0);
    iu::math::mulC(*in_0, w_2, *out_1);
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
