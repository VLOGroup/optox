///@file cutils.h
///@brief cuda utility functions
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 03.2019

#pragma once

#include "utils.h"

#include <cuda_runtime_api.h>
#include <cufft.h>

// includes for time measurements
#ifdef _WIN32
#include <time.h>
#include <windows.h>
#else
#include <sys/time.h>
#endif

class OpToXCudaException : public OpToXException
{
public:
  OpToXCudaException(const cudaError_t cudaErr,
                     const char *file = nullptr,
                     const char *function = nullptr,
                     int line = 0) throw() : OpToXException(std::string("CUDA Error: ") + cudaGetErrorString(cudaErr), file, function, line),
                                             cudaErr_(cudaErr)
  {
  }

protected:
  cudaError_t cudaErr_;
};

namespace optox
{

static inline void checkCudaErrorState(const char *file, const char *function, const int line)
{
#if defined(THROW_CUDA_ERROR)
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw OpToXCudaException(err, file, function, line);
#endif
}

static inline void checkCudaErrorState(cudaError_t err, const char *file, const char *function,
                                       const int line)
{
  if (cudaSuccess != err)
  {
    throw OpToXCudaException(err, file, function, line);
  }
}

class OpToXCufftException : public OpToXException
{
public:
  OpToXCufftException(const cufftResult cudaErr,
                      const char *file = nullptr,
                      const char *function = nullptr,
                      int line = 0) throw() : OpToXException(std::string("CUFFT Error: ") + cufftGetErrorString(cudaErr),
                                                             file, function, line),
                                              cufftResult_(cudaErr)
  {
  }

protected:
  cufftResult cufftResult_;

private:
  static const char *cufftGetErrorString(cufftResult err)
  {
    switch (err)
    {
    case CUFFT_SUCCESS:
      return "The cuFFT operation was successful.";
    case CUFFT_INVALID_PLAN:
      return "cuFFT was passed an invalid plan handle.";
    case CUFFT_ALLOC_FAILED:
      return "cuFFT failed to allocate GPU or CPU memory.";
    case CUFFT_INVALID_VALUE:
      return "User specified an invalid pointer or parameter.";
    case CUFFT_INTERNAL_ERROR:
      return "Driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED:
      return "Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED:
      return "The cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
      return "User specified an invalid transform size";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "Missing parameters in call";
    case CUFFT_INVALID_DEVICE:
      return "Execution of a plan was on different GPU than plan creation";
    case CUFFT_PARSE_ERROR:
      return "Internal plan database error";
    case CUFFT_NO_WORKSPACE:
      return "No workspace has been provided prior to plan execution";
    default:
      return "Unknown CUFFT error.";
    }
  }
};

static inline void checkCufftErrorState(const cufftResult status,
                                        const char *file, const char *function,
                                        const int line)
{
  if (status != CUFFT_SUCCESS)
    throw OpToXCufftException(status, file, function, line);
}

}

#define OPTOX_CUDA_CHECK optox::checkCudaErrorState(__FILE__, __FUNCTION__, __LINE__)
#define OPTOX_CUDA_SAFE_CALL(fun) optox::checkCudaErrorState(fun, __FILE__, __FUNCTION__, __LINE__)
#define OPTOX_CUFFT_SAFE_CALL(state) optox::checkCufftErrorState(state, __FILE__, __FUNCTION__, __LINE__)
