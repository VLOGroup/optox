///@file d_tensor.h
///@brief Device n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "utils.h"
#include "cutils.h"
#include "optox_api.h"
#include "tensor/tensor.h"

#include <string>
#include <type_traits>
#include <initializer_list>

#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

namespace optox
{

template <typename T, unsigned int N>
class HTensor;

template <typename T, unsigned int N>
class OPTOX_DLLAPI DTensor : public Tensor<N>
{
  private:
    T *data_;
    // flag to indicate if a deep copy occurd or not
    bool wrap_;

  public:
    typedef T type;

    DTensor() : Tensor<N>(), data_(nullptr), wrap_(false)
    {
    }

    DTensor(const Shape<N> &size) : Tensor<N>(size), data_(nullptr), wrap_(false)
    {
        OPTOX_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&data_), this->bytes()));
        if (data_ == nullptr)
            throw std::bad_alloc();
    }

    DTensor(T *data, const Shape<N> &size, bool wrap = false) : Tensor<N>(size), data_(nullptr), wrap_(wrap)
    {
        if (data == nullptr)
            THROW_OPTOXEXCEPTION("input data not valid");
        if (wrap_)
            data_ = data;
        else
        {
            OPTOX_CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&data_), this->bytes()));
            if (data_ == nullptr)
                throw std::bad_alloc();
            OPTOX_CUDA_SAFE_CALL(cudaMemcpy(this->data_, data, this->bytes(), cudaMemcpyDeviceToDevice));
        }
    }

    DTensor(DTensor const &) = delete;
    void operator=(DTensor const &) = delete;

    virtual ~DTensor()
    {
        if ((!wrap_) && (data_ != nullptr))
        {
            OPTOX_CUDA_SAFE_CALL(cudaFree(data_));
            data_ = nullptr;
        }
    }

    void copyFromHostPtr(const T *from)
    {
        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from, this->bytes(), cudaMemcpyHostToDevice));
    }

    void copyFrom(const DTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyDeviceToDevice));
    }

    void copyFrom(const HTensor<T, N> &from)
    {
        if (this->size() != from.size() || this->stride() != from.stride())
            THROW_OPTOXEXCEPTION("Invalid size or stride in copy!");

        OPTOX_CUDA_SAFE_CALL(
            cudaMemcpy(this->data_, from.constptr(), this->bytes(), cudaMemcpyHostToDevice));
    }

    #ifdef __CUDACC__
    template <typename T2>
    void fill(T2 scalar, const cudaStream_t stream = nullptr)
    {
        if (stream != nullptr)
            thrust::fill(thrust::cuda::par.on(stream), this->begin(), this->end(), static_cast<T>(scalar));
        else
            thrust::fill(this->begin(), this->end(), static_cast<T>(scalar));
    }
    #endif

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

    thrust::device_ptr<T> begin(void)
    {
        return thrust::device_ptr<T>(ptr());
    }

    thrust::device_ptr<T> end(void)
    {
        return thrust::device_ptr<T>(ptr() + this->numel());
    }

    const thrust::device_ptr<T> begin(void) const
    {
        return thrust::device_ptr<T>(constptr());
    }

    const thrust::device_ptr<T> end(void) const
    {
        return thrust::device_ptr<T>(constptr() + this->numel());
    }

    virtual bool onDevice() const
    {
        return true;
    }

    struct Ref
    {
      private:
        T *data_;
        const Shape<N> stride_;

      public:
        const Shape<N> size_;

        __HOST__ Ref(DTensor<T, N> &t) : data_(t.ptr()), stride_(t.stride()), size_(t.size())
        {
        }

        __HOST__ ~Ref()
        {
        }

        __DEVICE__ T &operator()(std::initializer_list<size_t> list)
        {
            return data_[computeIndex(list)];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 1>::value>::value>::type>
        __DEVICE__ T &operator()(T2 i)
        {
            static_assert(N == 1, "wrong access for 1dim Tensor");
            return data_[i * stride_[0]];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 2>::value>::value>::type>
        __DEVICE__ T &operator()(T2 i, T2 j)
        {
            static_assert(N == 2, "wrong access for 2dim Tensor");
            return data_[i * stride_[0] + j * stride_[1]];
        }

        template <typename T2, class = typename std::enable_if<
                                   std::integral_constant<bool,
                                                          std::is_integral<T2>::value &&
                                                              std::integral_constant<bool, N == 3>::value>::value>::type>
        __DEVICE__ T &operator()(T2 i, T2 j, T2 k)
        {
            static_assert(N == 3, "wrong access for 3dim Tensor");
            return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>::value>::type>
        __DEVICE__ T &operator()(A0 a0, Args... args)
        {
            static_assert(sizeof...(Args) == N - 1, "wrong access for Ndim Tensor");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        __DEVICE__ size_t computeIndex(const std::initializer_list<size_t> &list) const
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

        __HOST__ ConstRef(const DTensor<T, N> &t) : data_(t.constptr()), stride_(t.stride()), size_(t.size())
        {
        }

        __HOST__ ~ConstRef()
        {
        }

        __DEVICE__ const T &operator()(std::initializer_list<size_t> list) const
        {
            return data_[computeIndex(list)];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 1)>::type>
        __DEVICE__ const T &operator()(T2 i) const
        {
            return data_[i * stride_[0]];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 2)>::type>
        __DEVICE__ const T &operator()(T2 i, T2 j) const
        {
            return data_[i * stride_[0] + j * stride_[1]];
        }

        template <typename T2, class = typename std::enable_if<std::is_integral<T2>{} && (N == 3)>::type>
        __DEVICE__ const T &operator()(T2 i, T2 j, T2 k) const
        {
            return data_[i * stride_[0] + j * stride_[1] + k * stride_[2]];
        }

        template <typename A0, typename... Args, class = typename std::enable_if<std::is_integral<A0>{} && (N > 3)>::type>
        __DEVICE__ const T &operator()(A0 a0, Args... args) const
        {
            static_assert(sizeof...(Args) == N - 1, "size missmatch");
            return (*this)(std::initializer_list<size_t>({size_t(a0), size_t(args)...}));
        }

      private:
        __DEVICE__ size_t computeIndex(const std::initializer_list<size_t> &list) const
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
