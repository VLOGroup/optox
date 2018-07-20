///@file py_utils.h
///@brief utility functions for python to optox
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 09.07.2018

#pragma once

#define BOOST_PYTHON_STATIC_LIB

#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <iu/iupython.h>
#include <boost/python/overloads.hpp>
#include <boost/python/dict.hpp>

#include <memory>

#include "operators/ioperator.h"

namespace bp = boost::python;

void mapFromPyObject(const bp::object &py_ob, optox::OperatorConfigDict &out_map)
{
    bp::dict py_dict = bp::extract<bp::dict>(py_ob);
    boost::python::list keys = py_dict.keys();
    for (int i = 0; i < bp::len(keys); ++i)
    {
        std::string key = std::string(bp::extract<const char *>(keys[i]));
        bp::object py_val_fun = py_dict[key].attr("__str__");
        bp::object py_val = py_val_fun();
        std::string value = std::string(bp::extract<const char *>(py_val));
        out_map[key] = value;
    }
}

template <typename T, unsigned int N>
std::unique_ptr<iu::LinearDeviceMemory<T, N>> getLinearDeviceFromNumpy(bp::object &py_arr)
{
    iu::LinearHostMemory<T, N> host_mem(py_arr);
    std::unique_ptr<iu::LinearDeviceMemory<T, N>> p(new iu::LinearDeviceMemory<T, N>(host_mem.size()));
    iu::copy(&host_mem, p.get());

    // do not return a copy but rather move its value
    return move(p);
}

template <template <typename, unsigned int> class TOperator, typename T, unsigned int N>
void setConfig(bp::object &self, bp::object &py_ob)
{
    TOperator<T, N> &op = bp::extract<TOperator<T, N> &>(self);
    optox::OperatorConfigDict config;
    mapFromPyObject(py_ob, config);
    op.setConfig(config);
}
