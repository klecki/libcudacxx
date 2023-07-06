//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class Duration2>
//   time_point(const time_point<clock, Duration2>& t);

// Duration2 shall be implicitly convertible to duration.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <cuda_for_dali/std/chrono>

int main(int, char**)
{
    typedef cuda_for_dali::std::chrono::system_clock Clock;
    typedef cuda_for_dali::std::chrono::milliseconds Duration1;
    typedef cuda_for_dali::std::chrono::microseconds Duration2;
    {
    cuda_for_dali::std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    cuda_for_dali::std::chrono::time_point<Clock, Duration1> t1 = t2;
    }

  return 0;
}
