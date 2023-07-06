//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// typedef atomic<char>               atomic_char;
// typedef atomic<signed char>        atomic_schar;
// typedef atomic<unsigned char>      atomic_uchar;
// typedef atomic<short>              atomic_short;
// typedef atomic<unsigned short>     atomic_ushort;
// typedef atomic<int>                atomic_int;
// typedef atomic<unsigned int>       atomic_uint;
// typedef atomic<long>               atomic_long;
// typedef atomic<unsigned long>      atomic_ulong;
// typedef atomic<long long>          atomic_llong;
// typedef atomic<unsigned long long> atomic_ullong;
// typedef atomic<char16_t>           atomic_char16_t;
// typedef atomic<char32_t>           atomic_char32_t;
// typedef atomic<wchar_t>            atomic_wchar_t;
//
// typedef atomic<intptr_t>           atomic_intptr_t;
// typedef atomic<uintptr_t>          atomic_uintptr_t;
//
// typedef atomic<int8_t>             atomic_int8_t;
// typedef atomic<uint8_t>            atomic_uint8_t;
// typedef atomic<int16_t>            atomic_int16_t;
// typedef atomic<uint16_t>           atomic_uint16_t;
// typedef atomic<int32_t>            atomic_int32_t;
// typedef atomic<uint32_t>           atomic_uint32_t;
// typedef atomic<int64_t>            atomic_int64_t;
// typedef atomic<uint64_t>           atomic_uint64_t;

#include <cuda_for_dali/std/atomic>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<char>, cuda_for_dali::std::atomic_char>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<signed char>, cuda_for_dali::std::atomic_schar>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<unsigned char>, cuda_for_dali::std::atomic_uchar>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<short>, cuda_for_dali::std::atomic_short>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<unsigned short>, cuda_for_dali::std::atomic_ushort>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<int>, cuda_for_dali::std::atomic_int>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<unsigned int>, cuda_for_dali::std::atomic_uint>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<long>, cuda_for_dali::std::atomic_long>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<unsigned long>, cuda_for_dali::std::atomic_ulong>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<long long>, cuda_for_dali::std::atomic_llong>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<unsigned long long>, cuda_for_dali::std::atomic_ullong>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<wchar_t>, cuda_for_dali::std::atomic_wchar_t>::value), "");
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<char16_t>, cuda_for_dali::std::atomic_char16_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<char32_t>, cuda_for_dali::std::atomic_char32_t>::value), "");
#endif  // _LIBCUDACXX_HAS_NO_UNICODE_CHARS

//  Added by LWG 2441
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<intptr_t>,  cuda_for_dali::std::atomic_intptr_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<uintptr_t>, cuda_for_dali::std::atomic_uintptr_t>::value), "");

    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<int8_t>,    cuda_for_dali::std::atomic_int8_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<uint8_t>,   cuda_for_dali::std::atomic_uint8_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<int16_t>,   cuda_for_dali::std::atomic_int16_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<uint16_t>,  cuda_for_dali::std::atomic_uint16_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<int32_t>,   cuda_for_dali::std::atomic_int32_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<uint32_t>,  cuda_for_dali::std::atomic_uint32_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<int64_t>,   cuda_for_dali::std::atomic_int64_t>::value), "");
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::atomic<uint64_t>,  cuda_for_dali::std::atomic_uint64_t>::value), "");

  return 0;
}
