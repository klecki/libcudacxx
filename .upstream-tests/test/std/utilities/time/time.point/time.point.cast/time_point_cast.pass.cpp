//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// time_point

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   time_point_cast(const time_point<Clock, Duration>& t);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

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
    typedef decltype(cuda_for_dali::std::chrono::time_point_cast<ToDuration>(f)) R;
    static_assert((cuda_for_dali::std::is_same<R, ToTimePoint>::value), "");
    assert(cuda_for_dali::std::chrono::time_point_cast<ToDuration>(f) == t);
    }
}

#if TEST_STD_VER > 11

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
    static_assert(cuda_for_dali::std::chrono::time_point_cast<ToDuration>(f) == t, "");
    }

}

#endif

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
#if TEST_STD_VER > 11
    {
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::hours,    2> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::minutes,121> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::seconds,7265> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::milliseconds,7265000> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::microseconds,7265000000LL> ();
    test_constexpr<cuda_for_dali::std::chrono::milliseconds, 7265000, cuda_for_dali::std::chrono::nanoseconds,7265000000000LL> ();
    typedef cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<3, 5>> T1;
    test_constexpr<cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<2, 3>>, 9, T1, 10> ();
    }
#endif

  return 0;
}
