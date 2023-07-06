//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// typedef duration<signed integral type of at least 64 bits, nano> nanoseconds;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/limits>

int main(int, char**)
{
    typedef cuda_for_dali::std::chrono::nanoseconds D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(cuda_for_dali::std::is_signed<Rep>::value, "");
    static_assert(cuda_for_dali::std::is_integral<Rep>::value, "");
    static_assert(cuda_for_dali::std::numeric_limits<Rep>::digits >= 63, "");
    static_assert((cuda_for_dali::std::is_same<Period, cuda_for_dali::std::nano>::value), "");

  return 0;
}
