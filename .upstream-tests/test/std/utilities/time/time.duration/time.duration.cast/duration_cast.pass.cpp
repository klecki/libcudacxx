//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   duration_cast(const duration<Rep, Period>& d);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <class ToDuration, class FromDuration>
__host__ __device__
void
test(const FromDuration& f, const ToDuration& d)
{
    {
    typedef decltype(cuda_for_dali::std::chrono::duration_cast<ToDuration>(f)) R;
    static_assert((cuda_for_dali::std::is_same<R, ToDuration>::value), "");
    assert(cuda_for_dali::std::chrono::duration_cast<ToDuration>(f) == d);
    }
}

int main(int, char**)
{
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::hours(2));
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::minutes(121));
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::seconds(7265));
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::milliseconds(7265000));
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::microseconds(7265000000LL));
    test(cuda_for_dali::std::chrono::milliseconds(7265000), cuda_for_dali::std::chrono::nanoseconds(7265000000000LL));
    test(cuda_for_dali::std::chrono::milliseconds(7265000),
         cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3600> >(7265./3600));
    test(cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> >(9),
         cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> >(10));
#if TEST_STD_VER >= 11
    {
    constexpr cuda_for_dali::std::chrono::hours h = cuda_for_dali::std::chrono::duration_cast<cuda_for_dali::std::chrono::hours>(cuda_for_dali::std::chrono::milliseconds(7265000));
    static_assert(h.count() == 2, "");
    }
#endif

  return 0;
}
