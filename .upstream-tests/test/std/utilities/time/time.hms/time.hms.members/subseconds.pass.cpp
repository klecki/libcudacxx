//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

// template <class Duration>
// class hh_mm_ss
//
// constexpr precision subseconds() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <typename Duration>
__host__ __device__
constexpr auto check_subseconds(Duration d)
{
    using HMS = cuda_for_dali::std::chrono::hh_mm_ss<Duration>;
    ASSERT_SAME_TYPE(typename HMS::precision, decltype(cuda_for_dali::std::declval<HMS>().subseconds()));
    ASSERT_NOEXCEPT(                                   cuda_for_dali::std::declval<HMS>().subseconds());
    return HMS(d).subseconds().count();
}

int main(int, char**)
{
    using microfortnights = cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<756, 625>>;

    static_assert( check_subseconds(cuda_for_dali::std::chrono::seconds( 1)) == 0, "");
    static_assert( check_subseconds(cuda_for_dali::std::chrono::seconds(-1)) == 0, "");

    assert( check_subseconds(cuda_for_dali::std::chrono::seconds( 5000)) == 0);
    assert( check_subseconds(cuda_for_dali::std::chrono::seconds(-5000)) == 0);
    assert( check_subseconds(cuda_for_dali::std::chrono::minutes( 5000)) == 0);
    assert( check_subseconds(cuda_for_dali::std::chrono::minutes(-5000)) == 0);
    assert( check_subseconds(cuda_for_dali::std::chrono::hours( 11))     == 0);
    assert( check_subseconds(cuda_for_dali::std::chrono::hours(-11))     == 0);

    assert( check_subseconds(cuda_for_dali::std::chrono::milliseconds( 123456789LL)) == 789);
    assert( check_subseconds(cuda_for_dali::std::chrono::milliseconds(-123456789LL)) == 789);
    assert( check_subseconds(cuda_for_dali::std::chrono::microseconds( 123456789LL)) == 456789LL);
    assert( check_subseconds(cuda_for_dali::std::chrono::microseconds(-123456789LL)) == 456789LL);
    assert( check_subseconds(cuda_for_dali::std::chrono::nanoseconds( 123456789LL))  == 123456789LL);
    assert( check_subseconds(cuda_for_dali::std::chrono::nanoseconds(-123456789LL))  == 123456789LL);

    assert( check_subseconds(microfortnights(  1000)) == 6000);
    assert( check_subseconds(microfortnights( -1000)) == 6000);
    assert( check_subseconds(microfortnights( 10000)) == 0);
    assert( check_subseconds(microfortnights(-10000)) == 0);

    return 0;
}
