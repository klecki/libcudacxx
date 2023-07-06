//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// using weeks = duration<signed integer type of at least 22 bits, ratio_multiply<ratio<7>, days::period>>;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/limits>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda_for_dali::std::chrono::weeks D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(cuda_for_dali::std::is_signed<Rep>::value, "");
    static_assert(cuda_for_dali::std::is_integral<Rep>::value, "");
    static_assert(cuda_for_dali::std::numeric_limits<Rep>::digits >= 22, "");
    static_assert(cuda_for_dali::std::is_same_v<Period, cuda_for_dali::std::ratio_multiply<cuda_for_dali::std::ratio<7>, cuda_for_dali::std::chrono::days::period>>, "");

  return 0;
}
