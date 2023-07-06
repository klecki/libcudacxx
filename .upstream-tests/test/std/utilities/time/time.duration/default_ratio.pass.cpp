//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// Test default template arg:

// template <class Rep, class Period = ratio<1>>
// class duration;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>

int main(int, char**)
{
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1> >,
                   cuda_for_dali::std::chrono::duration<int> >::value), "");

  return 0;
}
