//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month_day;

//  explicit constexpr year_month_day(const local_days& dp) noexcept;
//
//
//  Effects:  Constructs an object of type year_month_day that corresponds
//                to the date represented by dp
//
//  Remarks: Equivalent to constructing with sys_days{dp.time_since_epoch()}.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year           = cuda_for_dali::std::chrono::year;
    using day            = cuda_for_dali::std::chrono::day;
    using local_days     = cuda_for_dali::std::chrono::local_days;
    using days           = cuda_for_dali::std::chrono::days;
    using year_month_day = cuda_for_dali::std::chrono::year_month_day;

    ASSERT_NOEXCEPT(year_month_day{cuda_for_dali::std::declval<local_days>()});

    auto constexpr January = cuda_for_dali::std::chrono::January;

    {
    constexpr local_days sd{};
    constexpr year_month_day ymd{sd};

    static_assert( ymd.ok(),                            "");
    static_assert( ymd.year()  == year{1970},           "");
    static_assert( ymd.month() == January, "");
    static_assert( ymd.day()   == day{1},               "");
    }

    {
    constexpr local_days sd{days{10957+32}};
    constexpr year_month_day ymd{sd};

    auto constexpr February = cuda_for_dali::std::chrono::February;

    static_assert( ymd.ok(),                             "");
    static_assert( ymd.year()  == year{2000},            "");
    static_assert( ymd.month() == February, "");
    static_assert( ymd.day()   == day{2},                "");
    }


//  There's one more leap day between 1/1/40 and 1/1/70
//  when compared to 1/1/70 -> 1/1/2000
    {
    constexpr local_days sd{days{-10957}};
    constexpr year_month_day ymd{sd};

    static_assert( ymd.ok(),                            "");
    static_assert( ymd.year()  == year{1940},           "");
    static_assert( ymd.month() == January, "");
    static_assert( ymd.day()   == day{2},               "");
    }

    {
    local_days sd{days{-(10957+34)}};
    year_month_day ymd{sd};
    auto constexpr November = cuda_for_dali::std::chrono::November;

    assert( ymd.ok());
    assert( ymd.year()  == year{1939});
    assert( ymd.month() == November);
    assert( ymd.day()   == day{29});
    }

  return 0;
}
