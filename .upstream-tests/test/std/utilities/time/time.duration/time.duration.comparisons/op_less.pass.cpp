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
//   constexpr
//   bool
//   operator< (const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator> (const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator<=(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

// template <class Rep1, class Period1, class Rep2, class Period2>
//   constexpr
//   bool
//   operator>=(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    cuda_for_dali::std::chrono::seconds s1(3);
    cuda_for_dali::std::chrono::seconds s2(3);
    assert(!(s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert( (s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::seconds s1(3);
    cuda_for_dali::std::chrono::seconds s2(4);
    assert( (s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert(!(s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::milliseconds s1(3);
    cuda_for_dali::std::chrono::microseconds s2(3000);
    assert(!(s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert( (s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::milliseconds s1(3);
    cuda_for_dali::std::chrono::microseconds s2(4000);
    assert( (s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert(!(s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(9);
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(10);
    assert(!(s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert( (s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(10);
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(9);
    assert(!(s1 < s2));
    assert( (s1 > s2));
    assert(!(s1 <= s2));
    assert( (s1 >= s2));
    }
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(9);
    cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3, 5> > s2(10);
    assert(!(s1 < s2));
    assert(!(s1 > s2));
    assert( (s1 <= s2));
    assert( (s1 >= s2));
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda_for_dali::std::chrono::seconds s1(3);
    constexpr cuda_for_dali::std::chrono::seconds s2(3);
    static_assert(!(s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert( (s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::seconds s1(3);
    constexpr cuda_for_dali::std::chrono::seconds s2(4);
    static_assert( (s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert(!(s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::milliseconds s1(3);
    constexpr cuda_for_dali::std::chrono::microseconds s2(3000);
    static_assert(!(s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert( (s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::milliseconds s1(3);
    constexpr cuda_for_dali::std::chrono::microseconds s2(4000);
    static_assert( (s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert(!(s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(9);
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(10);
    static_assert(!(s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert( (s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(10);
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5> > s2(9);
    static_assert(!(s1 < s2), "");
    static_assert( (s1 > s2), "");
    static_assert(!(s1 <= s2), "");
    static_assert( (s1 >= s2), "");
    }
    {
    constexpr cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3> > s1(9);
    constexpr cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3, 5> > s2(10);
    static_assert(!(s1 < s2), "");
    static_assert(!(s1 > s2), "");
    static_assert( (s1 <= s2), "");
    static_assert( (s1 >= s2), "");
    }
#endif

  return 0;
}
