//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// duration& operator-=(const duration& d);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 14
__host__ __device__
constexpr bool test_constexpr()
{
    cuda_for_dali::std::chrono::seconds s(3);
    s -= cuda_for_dali::std::chrono::seconds(2);
    if (s.count() != 1) return false;
    s -= cuda_for_dali::std::chrono::minutes(2);
    return s.count() == -119;
}
#endif

int main(int, char**)
{
    {
    cuda_for_dali::std::chrono::seconds s(3);
    s -= cuda_for_dali::std::chrono::seconds(2);
    assert(s.count() == 1);
    s -= cuda_for_dali::std::chrono::minutes(2);
    assert(s.count() == -119);
    }

#if TEST_STD_VER > 14
    static_assert(test_constexpr(), "");
#endif

  return 0;
}
