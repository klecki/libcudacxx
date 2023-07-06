//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator+(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    cuda_for_dali::std::chrono::seconds s1(3);
    cuda_for_dali::std::chrono::seconds s2(5);
    cuda_for_dali::std::chrono::seconds r = s1 + s2;
    assert(r.count() == 8);
    }
    {
    cuda_for_dali::std::chrono::seconds s1(3);
    cuda_for_dali::std::chrono::microseconds s2(5);
    cuda_for_dali::std::chrono::microseconds r = s1 + s2;
    assert(r.count() == 3000005);
    }
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(3);
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(5);
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1, 15> > r = s1 + s2;
    assert(r.count() == 75);
    }
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(3);
    cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3, 5> > s2(5);
    cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<1, 15> > r = s1 + s2;
    assert(r.count() == 75);
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda_for_dali::std::chrono::seconds s1(3);
    constexpr cuda_for_dali::std::chrono::seconds s2(5);
    constexpr cuda_for_dali::std::chrono::seconds r = s1 + s2;
    static_assert(r.count() == 8, "");
    }
    {
    constexpr cuda_for_dali::std::chrono::seconds s1(3);
    constexpr cuda_for_dali::std::chrono::microseconds s2(5);
    constexpr cuda_for_dali::std::chrono::microseconds r = s1 + s2;
    static_assert(r.count() == 3000005, "");
    }
    {
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(3);
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(5);
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1, 15> > r = s1 + s2;
    static_assert(r.count() == 75, "");
    }
    {
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(3);
    constexpr cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3, 5> > s2(5);
    constexpr cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<1, 15> > r = s1 + s2;
    static_assert(r.count() == 75, "");
    }
#endif

  return 0;
}
