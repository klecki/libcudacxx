//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_for_dali/std/ctime>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

#ifndef __CUDACC_RTC__
#ifndef CLOCKS_PER_SEC
#error CLOCKS_PER_SEC not defined
#endif
#endif

#if TEST_STD_VER > 14 && defined(TEST_HAS_C11_FEATURES)
#ifndef TIME_UTC
#error TIME_UTC not defined
#endif
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wformat-zero-length"
#endif

#pragma nv_diag_suppress set_but_not_used

int main(int, char**)
{
    cuda_for_dali::std::clock_t c = 0;
    cuda_for_dali::std::size_t s = 0;
    cuda_for_dali::std::time_t t = 0;
    ((void)c); // Prevent unused warning
    ((void)s); // Prevent unused warning
    ((void)t); // Prevent unused warning
#ifndef __CUDACC_RTC__
    cuda_for_dali::std::tm tm = {};
    char str[3];
    ((void)tm); // Prevent unused warning
    ((void)str); // Prevent unused warning
#if TEST_STD_VER > 14 && defined(TEST_HAS_C11_FEATURES)
    cuda_for_dali::std::timespec tmspec = {};
    ((void)tmspec); // Prevent unused warning
#endif

    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::clock()), cuda_for_dali::std::clock_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::difftime(t,t)), double>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::mktime(&tm)), cuda_for_dali::std::time_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::time(&t)), cuda_for_dali::std::time_t>::value), "");
#if TEST_STD_VER > 14 && defined(TEST_HAS_TIMESPEC_GET)
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::timespec_get(&tmspec, 0)), int>::value), "");
#endif
#ifndef _LIBCUDACXX_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::asctime(&tm)), char*>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::ctime(&t)), char*>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::gmtime(&t)), cuda_for_dali::std::tm*>::value), "");
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::localtime(&t)), cuda_for_dali::std::tm*>::value), "");
#endif
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::strftime(str,s,"",&tm)), cuda_for_dali::std::size_t>::value), "");
#endif

  return 0;
}
