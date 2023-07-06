//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year = cuda_for_dali::std::chrono::year;

    static_assert(cuda_for_dali::std::is_trivially_copyable_v<year>, "");
    static_assert(cuda_for_dali::std::is_standard_layout_v<year>, "");

  return 0;
}
