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
// XFAIL: c++98, c++03

// <cuda/std/atomic>

// #define ATOMIC_VAR_INIT(value)

#include <cuda_for_dali/std/atomic>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    cuda_for_dali::std::atomic<int> v = ATOMIC_VAR_INIT(5);
    assert(v == 5);

  return 0;
}
