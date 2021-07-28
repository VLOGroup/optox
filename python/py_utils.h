///@file py_utils.h
///@brief utility functions for python to optox
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tensor/h_tensor.h>
#include <tensor/d_tensor.h>

#include "typetraits.h"

namespace py = pybind11;

template <typename T, unsigned int N>
std::unique_ptr<optox::DTensor<T, N>> getDTensorNp(py::array &array)
{
    if (array.ndim() != N)
        throw std::runtime_error("invalid tensor dimensions. expected " + std::to_string(N) +
                                 " but got " + std::to_string(array.ndim()) + "!");

    py::buffer_info info = array.request();
    if (info.format != py::format_descriptor<T>::format())
        throw std::runtime_error("invalid tensor dtype. expected " +
                                 py::format_descriptor<T>::format() + " but got " +
                                 info.format + "!");

    // wrap the Tensor into a device tensor
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
        size[i] = array.shape()[i];
    std::unique_ptr<optox::DTensor<T, N>> p(new optox::DTensor<T, N>(size));
    p->copyFromHostPtr(reinterpret_cast<const T *>(array.data()));

    return p;
}

template <typename T, unsigned int N>
std::unique_ptr<optox::DTensor<T, N>> getComplexDTensorNp(py::array &array)
{
    if (array.ndim() != N + 1)
        throw std::runtime_error("invalid tensor dimensions. expected " + std::to_string(N + 1) +
                                 " but got " + std::to_string(array.ndim()) + "!");

    //typedef typename optox::type_trait<T>::complex_type complex_type;
    typedef typename optox::type_trait<T>::real_type real_type;

    py::buffer_info info = array.request();
    if (info.format != py::format_descriptor<real_type>::format())
        throw std::runtime_error("invalid tensor dtype. expected " +
                                 py::format_descriptor<real_type>::format() + " but got " +
                                 info.format + "!");
    
    optox::Shape<N> size;
    for (unsigned int i = 0; i < N; ++i)
    {
        size[i] = array.shape()[i];
    }
    std::unique_ptr<optox::DTensor<T, N>> p(new optox::DTensor<T, N>(size));
    p->copyFromHostPtr(reinterpret_cast<const T *>(array.data()));

    return move(p);
}

template <typename T, unsigned int N>
py::array dTensorToNp(const optox::DTensor<T, N> &d_tensor)
{
    // transfer to host
    optox::HTensor<T, N> h_tensor(d_tensor.size());
    h_tensor.copyFrom(d_tensor);

    ssize_t ndim = N;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    for (unsigned int i = 0; i < N; ++i)
    {
        shape.push_back(h_tensor.size()[i]);
        // stride needs to be in bytes
        strides.push_back(h_tensor.stride()[i] * sizeof(T));
    }

    return py::array(py::buffer_info(
        h_tensor.ptr(),
        sizeof(T),
        py::format_descriptor<T>::format(),
        ndim,
        shape,
        strides));
}

template <typename T, unsigned int N>
py::array dComplexTensorToNp(const optox::DTensor<T, N> &d_tensor)
{
    // transfer to host
    optox::HTensor<T, N> h_tensor(d_tensor.size());
    h_tensor.copyFrom(d_tensor);

    ssize_t ndim = N + 1;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    for (unsigned int i = 0; i < N; ++i)
    {
        shape.push_back(h_tensor.size()[i]);
        // stride needs to be in bytes
        strides.push_back(2 * h_tensor.stride()[i] * sizeof(typename optox::type_trait<T>::real_type ));
    }
    shape.push_back(2);
    strides.push_back(1 * sizeof(typename optox::type_trait<T>::real_type));

    return py::array(py::buffer_info(
        reinterpret_cast<typename optox::type_trait<T>::real_type *>(h_tensor.ptr()),
        sizeof(typename optox::type_trait<T>::real_type),
        py::format_descriptor<typename optox::type_trait<T>::real_type>::format(),
        ndim,
        shape,
        strides));
}
