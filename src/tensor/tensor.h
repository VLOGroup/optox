///@file tensor.h
///@brief Basic n-dimensional tensor class for library
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "optox_api.h"
#include "tensor/shape.h"

namespace optox
{

class OPTOX_DLLAPI ITensor
{
  public:
    ITensor()
    {
    }

    virtual ~ITensor()
    {
    }

    ITensor(ITensor const &) = delete;
    void operator=(ITensor const &) = delete;

    virtual bool onDevice() const = 0;

    virtual size_t bytes() const = 0;
};

template <unsigned int N>
class OPTOX_DLLAPI Tensor : public ITensor
{
  protected:
    Shape<N> size_;
    Shape<N> stride_;

    void computeStride()
    {
        stride_[0] = 1;
        for (unsigned int i = 1; i < N; i++)
            stride_[i] = stride_[i - 1] * size_[i - 1];
    }

  public:
    Tensor() : ITensor(), size_(), stride_()
    {
    }

    Tensor(const Shape<N> &size) : ITensor(), size_(size)
    {
        computeStride();
    }

    virtual ~Tensor()
    {
    }

    Tensor(Tensor const &) = delete;
    void operator=(Tensor const &) = delete;

    size_t numel() const
    {
        return size_.numel();
    }

    const Shape<N> &size() const
    {
        return size_;
    }

    const Shape<N> &stride() const
    {
        return stride_;
    }

    friend std::ostream &operator<<(std::ostream &out, Tensor const &t)
    {
        out << "Tensor: size=" << t.size() << " strides="
            << t.stride() << " numel=" << t.numel() << " onDevice="
            << t.onDevice();
        return out;
    }
};

} // namespace optox
