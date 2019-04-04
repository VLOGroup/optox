///@file shape.h
///@brief Basic shape class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "optox_api.h"
#include "utils.h"

#include <cassert>
#include <type_traits>
#include <vector>
#include <initializer_list>

namespace optox
{

template <unsigned int N>
struct OPTOX_DLLAPI Shape
{
  public:
    typedef size_t array_type[N];

  protected:
    array_type v;

  public:
    // default constructor
    __DHOSTDEVICE__ Shape() = default;

    // default copy constructor
    __DHOSTDEVICE__ Shape(const Shape<N> &) = default;

    // constructor from standard initializer list
    template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    __HOSTDEVICE__ Shape(std::initializer_list<T> list)
    {
        auto a = list.begin();
        for (unsigned int i = 0; i < N; ++i)
        {
            v[i] = static_cast<size_t>(*a);
            ++a;
        }
    }

    // variable argument constructor
    template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>::value>::type>
    __HOSTDEVICE__ Shape(A0 a0, Args... args) : Shape(std::initializer_list<size_t>({size_t(a0), size_t(args)...}))
    {
        static_assert(sizeof...(Args) == N - 1, "size missmatch");
    }

    // constructor from vector
    template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    __HOSTDEVICE__ Shape(const std::vector<T> &v) : Shape(std::initializer_list<size_t>(v.data(), v.data() + v.size()))
    {
        assert (v.size() == N);
    }

    // default operator =
    __DHOSTDEVICE__ Shape<N> &operator=(const Shape<N> &x) = default;

    // [] access operator
    template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    __HOSTDEVICE__ size_t &operator[](T i)
    {
        return v[i];
    }
    template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
    __HOSTDEVICE__ const size_t &operator[](T i) const
    {
        return v[i];
    }

    // shape comparison
    __HOSTDEVICE__ bool operator==(const Shape<N> &x) const
    {
        for (unsigned int i = 0; i < N; ++i)
        {
            if ((*this)[i] != x[i])
                return false;
        }
        return true;
    }
    __HOSTDEVICE__ bool operator!=(const Shape<N> &x) const
    {
        return !(*this == x);
    }

    __HOSTDEVICE__ size_t numel() const
    {
        size_t num = 1;
        for (unsigned int i = 0; i < N; ++i)
            num *= v[i];
        return num;
    }

    friend std::ostream &operator<<(std::ostream &out, Shape const &s)
    {
        out << "[";
        for (unsigned int i = 0; i < N - 1; i++)
            out << s.v[i] << ", ";
        out << s.v[N - 1] << "]";
        return out;
    }
};

} // namespace optox
