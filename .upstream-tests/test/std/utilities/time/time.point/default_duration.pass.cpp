//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// Test default template arg:

// template <class Clock, class Duration = typename Clock::duration>
//   class time_point;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>

int main(int, char**)
{
    static_assert((cuda_for_dali::std::is_same<cuda_for_dali::std::chrono::system_clock::duration,
                   cuda_for_dali::std::chrono::time_point<cuda_for_dali::std::chrono::system_clock>::duration>::value), "");

  return 0;
}
