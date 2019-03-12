///@file h_tensor.h
///@brief Host n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "cutils.h"
#include "optox_api.h"
#include "tensor/tensor.h"

#include <string>
#include <type_traits>
#include <initializer_list>

#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>

namespace optox
{

template <typename T, unsigned int N>
class DTensor;

template <typename T, unsigned int N>
class OPTOX_DLLAPI HTensor : public Tensor<N>
{
  private:
    T *data_;
    // flag to indicate if a deep copy occurd or not
    bool wrap_;

  public:
    typedef T type;

    HTensor() : Tensor<N>(), data_(nullptr), wrap_(false)
    {
    }

    HTensor(const Shape<N> &size) : Tensor<N>(size), data_(nullptr), wrap_(false)
    {
        data_ = new T[this->numel()];
        if (data_ == nullptr)
            throw std::bad_alloc();
    }

    HTensor(T *data, const Shape<N> &size, bool wrap = false) : Tensor<N>(size), data_(nullptr), wrap_(wrap)
    {
        if (data == nullptr)
            THROW_OPTOXEXCEPTION("input data not valid");
        if (wrap_)
            data_ = data;
        else
        {
            data_ = new T[this->numel()];
            if (data_ == nullptr)
                throw std::bad_alloc();
            memcpy(data_, data, this->bytes());
        }
    }

    HTensor(HTensor const &) = delete;
    void operator=(HTensor const &) = delete;

    virtual ~HTensor()
    {
        if ((!wrap_) && (data_ != nullptr))
        {
            delete[] data_;
            data_ = nullptr;
        }
    }

    void copyFrom(const HTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        memcpy(this->data_, from.constptr(), this->bytes());
    }

    void copyFrom(const DTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyDeviceToHost));
    }

    template <typename T2>
    void fill(T2 scalar)
    {
        thrust::fill(this->begin(), this->end(), static_cast<T>(scalar));
    }

    size_t bytes() const
    {
        return this->numel() * sizeof(T);
    }

    T *ptr(unsigned int offset = 0)
    {
        if (offset >= this->numel())
        {
            std::stringstream msg;
            msg << "Index (" << offset << ") out of range (" << this->numel() << ").";
            THROW_OPTOXEXCEPTION(msg.str());
        }
        return &(data_[offset]);
    }

    const T *constptr(unsigned int offset = 0) const
    {
        if (offset >= this->numel())
        {
            std::stringstream msg;
            msg << "Offset (" << offset << ") out of range (" << this->numel() << ").";
            THROW_OPTOXEXCEPTION(msg.str());
        }
        return reinterpret_cast<const T *>(&(data_[offset]));
    }

    thrust::pointer<T, thrust::host_system_tag> begin(void)
    {
        return thrust::pointer<T, thrust::host_system_tag>(ptr());
    }

    thrust::pointer<T, thrust::host_system_tag> end(void)
    {
        return thrust::pointer<T, thrust::host_system_tag>(ptr() + this->numel());
    }

    const thrust::pointer<const T, thrust::host_system_tag> begin(void) const
    {
        return thrust::pointer<const T, thrust::host_system_tag>(constptr());
    }

    const thrust::pointer<const T, thrust::host_system_tag> end(void) const
    {
        return thrust::pointer<const T, thrust::host_system_tag>(constptr() + this->numel());
    }

    virtual bool onDevice() const
    {
        return false;
    }

    struct Ref
    {
      private:
        T *data_;
        const Shape<N> stride_;

      public:
        const Shape<N> size_;

        Ref(HTensor<T, N> &t) : data_(t.ptr()), stride_(t.stride()), size_(t.size())
        {
        }

        ~Ref()
        {
        }

        T &operator()(std::initializer_list<size_t> list)
        {
            return data_[computeIndex(list)];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 1>::value>::value>::type>
        T &operator()(T2 i)
        {
            static_assert(N == 1, "wrong access for 1dim Tensor");
            return data_[i * stride_[0]];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 2>::value>::value>::type>
        T &operator()(T2 i, T2 j)
        {
            static_assert(N == 2, "wrong access for 2dim Tensor");
            return data_[i * stride_[0] + j * stride_[1]];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 3>::value>::value>::type>
        T &operator()(T2 i, T2 j, T2 k)
        {
            static_assert(N == 3, "wrong access for 3dim Tensor");
            return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>::value>::type>
        T &operator()(A0 a0, Args... args)
        {
            static_assert(sizeof...(Args) == N - 1, "wrong access for Ndim Tensor");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        size_t computeIndex(const std::initializer_list<size_t> &list) const
        {
            size_t idx = 0;
            auto e = list.begin();
            for (unsigned int i = 0; i < N; ++i)
            {
                idx += stride_[i] * (*e);
                ++e;
            }
            return idx;
        }
    };

    struct ConstRef
    {
      private:
        const T *data_;
        const Shape<N> stride_;

      public:
        const Shape<N> size_;

        ConstRef(const HTensor<T, N> &t) : data_(t.constptr()), stride_(t.stride()), size_(t.size())
        {
        }

        ~ConstRef()
        {
        }

        const T &operator()(std::initializer_list<size_t> list) const
        {
            return data_[computeIndex(list)];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
        const T &operator()(T2 i) const
        {
            return data_[i * stride_[0]];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
        const T &operator()(T2 i, T2 j) const
        {
            return data_[i * stride_[0] + j * stride_[1]];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
        const T &operator()(T2 i, T2 j, T2 k) const
        {
            return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{} && (N > 3)>::type>
        const T &operator()(A0 a0, Args... args) const
        {
            static_assert(sizeof...(Args) == N - 1, "size missmatch");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        size_t computeIndex(const std::initializer_list<size_t> &list) const
        {
            size_t idx = 0;
            auto e = list.begin();
            for (unsigned int i = 0; i < N; ++i)
            {
                idx += stride_[i] * (*e);
                ++e;
            }
            return idx;
        }
    };
};

} // namespace optox
