//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// system_clock

// time_t to_time_t(const time_point& t);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/ctime>

int main(int, char**)
{
    typedef cuda_for_dali::std::chrono::system_clock C;
    cuda_for_dali::std::time_t t1 = C::to_time_t(C::now());
    ((void)t1);

  return 0;
}
