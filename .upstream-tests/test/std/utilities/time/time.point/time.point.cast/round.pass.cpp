//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

// round

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   round(const time_point<Clock, Duration>& t);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

template <class FromDuration, class ToDuration>
__host__ __device__
void
test(const FromDuration& df, const ToDuration& d)
{
    typedef cuda_for_dali::std::chrono::system_clock Clock;
    typedef cuda_for_dali::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
    typedef cuda_for_dali::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
    {
    FromTimePoint f(df);
    ToTimePoint t(d);
    typedef decltype(cuda_for_dali::std::chrono::round<ToDuration>(f)) R;
    static_assert((cuda_for_dali::std::is_same<R, ToTimePoint>::value), "");
    assert(cuda_for_dali::std::chrono::round<ToDuration>(f) == t);
    }
}

template<class FromDuration, long long From, class ToDuration, long long To>
__host__ __device__
void test_constexpr ()
{
    typedef cuda_for_dali::std::chrono::system_clock Clock;
    typedef cuda_for_dali::std::chrono::time_point<Clock, FromDuration> FromTimePoint;
    typedef cuda_for_dali::std::chrono::time_point<Clock, ToDuration> ToTimePoint;
    {
    constexpr FromTimePoint f{FromDuration{From}};
    constexpr ToTimePoint   t{ToDuration{To}};
    static_assert(cuda_for_dali::std::chrono::round<ToDuration>(f) == t, "");
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(cuda_for_dali::std::chrono::milliseconds( 7290000), cuda_for_dali::std::chrono::hours( 2));
    test(cuda_for_dali::std::chrono::milliseconds(-7290000), cuda_for_dali::std::chrono::hours(-2));
    test(cuda_for_dali::std::chrono::milliseconds( 7290000), cuda_for_dali::std::chrono::minutes( 122));
    test(cuda_for_dali::std::chrono::milliseconds(-7290000), cuda_for_dali::std::chrono::minutes(-122));

//  9000000ms is 2 hours and 30 minutes
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 9000000, cuda_for_dali::std::chrono::hours,    2> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds,-9000000, cuda_for_dali::std::chrono::hours,   -2> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 9000001, cuda_for_dali::std::chrono::minutes, 150> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds,-9000001, cuda_for_dali::std::chrono::minutes,-150> ();

    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 9000000, cuda_for_dali::std::chrono::seconds, 9000> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds,-9000000, cuda_for_dali::std::chrono::seconds,-9000> ();

  return 0;
}
