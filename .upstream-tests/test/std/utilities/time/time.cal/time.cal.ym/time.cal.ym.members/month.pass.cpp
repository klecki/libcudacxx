//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class year_month;

// constexpr chrono::month month() const noexcept;
//  Returns: wd_

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year       = cuda_for_dali::std::chrono::year;
    using month      = cuda_for_dali::std::chrono::month;
    using year_month = cuda_for_dali::std::chrono::year_month;

    ASSERT_NOEXCEPT(                 std::declval<const year_month>().month());
    ASSERT_SAME_TYPE(month, decltype(cuda_for_dali::std::declval<const year_month>().month()));

    static_assert( year_month{}.month() == month{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month ym(year{1234}, month{i});
        assert( static_cast<unsigned>(ym.month()) == i);
    }

  return 0;
}
