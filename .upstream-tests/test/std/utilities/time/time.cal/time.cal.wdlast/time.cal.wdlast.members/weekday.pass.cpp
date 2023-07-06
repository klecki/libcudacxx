//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>
// class weekday_last;

//  constexpr chrono::weekday weekday() const noexcept;
//  Returns: wd_

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday      = cuda_for_dali::std::chrono::weekday;
    using weekday_last = cuda_for_dali::std::chrono::weekday_last;

    ASSERT_NOEXCEPT(                   std::declval<const weekday_last>().weekday());
    ASSERT_SAME_TYPE(weekday, decltype(cuda_for_dali::std::declval<const weekday_last>().weekday()));

    for (unsigned i = 0; i <= 255; ++i)
        assert(weekday_last{weekday{i}}.weekday() == weekday{i});

  return 0;
}
