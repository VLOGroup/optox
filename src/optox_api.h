///@file optox_api.h
///@brief Library API
///@author Erich Kobler <erich.kobler@icg.tugraz.at>
///@date 07.08.2018

#pragma once

/** Shared lib macros for windows dlls
*/
#ifdef WIN32
  #pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
  #pragma warning( disable : 4231 ) // disable the warning about nonstandard extension in e.g. istream

  #ifdef OPTOX_USE_STATIC
    #define OPTOX_DLLAPI
  #else
    #ifdef OPTOX_EXPORTS
      #define OPTOX_DLLAPI __declspec(dllexport)
    #else
      #define OPTOX_DLLAPI __declspec(dllimport)
    #endif
  #endif
#else
  #define OPTOX_DLLAPI
#endif


