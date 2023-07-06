//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"

// size_t should:

//  1. be in namespace std.
//  2. be the same sizeof as void*.
//  3. be an unsigned integral.

int main(int, char**)
{
    static_assert(sizeof(cuda_for_dali::std::size_t) == sizeof(void*),
                  "sizeof(cuda_for_dali::std::size_t) == sizeof(void*)");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::size_t>::value,
                  "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::size_t>::value");
    static_assert(cuda_for_dali::std::is_integral<cuda_for_dali::std::size_t>::value,
                  "cuda_for_dali::std::is_integral<cuda_for_dali::std::size_t>::value");

  return 0;
}
