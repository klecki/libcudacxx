//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_weekday;

// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const months& dm) noexcept;
//   Returns: ymwd + (-dm).
//
// constexpr year_month_weekday operator-(const year_month_weekday& ymwd, const years& dy) noexcept;
//   Returns: ymwd + (-dy).


#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"


__host__ __device__
constexpr bool testConstexprYears ()
{
    cuda_for_dali::std::chrono::year_month_weekday ym0{cuda_for_dali::std::chrono::year{1234}, cuda_for_dali::std::chrono::January, cuda_for_dali::std::chrono::weekday_indexed{cuda_for_dali::std::chrono::Tuesday, 1}};
    cuda_for_dali::std::chrono::year_month_weekday ym1 = ym0 - cuda_for_dali::std::chrono::years{10};
    return
        ym1.year()    == cuda_for_dali::std::chrono::year{1234-10}
     && ym1.month()   == cuda_for_dali::std::chrono::January
     && ym1.weekday() == cuda_for_dali::std::chrono::Tuesday
     && ym1.index()   == 1
        ;
}

__host__ __device__
constexpr bool testConstexprMonths ()
{
    cuda_for_dali::std::chrono::year_month_weekday ym0{cuda_for_dali::std::chrono::year{1234}, cuda_for_dali::std::chrono::November, cuda_for_dali::std::chrono::weekday_indexed{cuda_for_dali::std::chrono::Tuesday, 1}};
    cuda_for_dali::std::chrono::year_month_weekday ym1 = ym0 - cuda_for_dali::std::chrono::months{6};
    return
        ym1.year()    == cuda_for_dali::std::chrono::year{1234}
     && ym1.month()   == cuda_for_dali::std::chrono::May
     && ym1.weekday() == cuda_for_dali::std::chrono::Tuesday
     && ym1.index()   == 1
        ;
}


int main(int, char**)
{
    using year               = cuda_for_dali::std::chrono::year;
    using month              = cuda_for_dali::std::chrono::month;
    using weekday            = cuda_for_dali::std::chrono::weekday;
    using weekday_indexed    = cuda_for_dali::std::chrono::weekday_indexed;
    using year_month_weekday = cuda_for_dali::std::chrono::year_month_weekday;
    using years              = cuda_for_dali::std::chrono::years;
    using months             = cuda_for_dali::std::chrono::months;

    constexpr month November  = cuda_for_dali::std::chrono::November;
    constexpr weekday Tuesday = cuda_for_dali::std::chrono::Tuesday;

    {  // year_month_weekday - years
    ASSERT_NOEXCEPT(                              std::declval<year_month_weekday>() - std::declval<years>());
    ASSERT_SAME_TYPE(year_month_weekday, decltype(cuda_for_dali::std::declval<year_month_weekday>() - std::declval<years>()));

    static_assert(testConstexprYears(), "");

    year_month_weekday ym{year{1234}, November, weekday_indexed{Tuesday, 1}};
    for (int i = 0; i <= 10; ++i)
    {
        year_month_weekday ym1 = ym - years{i};
        assert(static_cast<int>(ym1.year()) == 1234 - i);
        assert(ym1.month()   == November);
        assert(ym1.weekday() == Tuesday);
        assert(ym1.index()   == 1);
    }
    }

    {  // year_month_weekday - months
    ASSERT_NOEXCEPT(                              std::declval<year_month_weekday>() - std::declval<months>());
    ASSERT_SAME_TYPE(year_month_weekday, decltype(cuda_for_dali::std::declval<year_month_weekday>() - std::declval<months>()));

    static_assert(testConstexprMonths(), "");

    year_month_weekday ym{year{1234}, November, weekday_indexed{Tuesday, 2}};
    for (unsigned i = 1; i <= 10; ++i)
    {
        year_month_weekday ym1 = ym - months{i};
        assert(ym1.year()    == year{1234});
        assert(ym1.month()   == month{11-i});
        assert(ym1.weekday() == Tuesday);
        assert(ym1.index()   == 2);
    }
    }

  return 0;
}
