//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// inline constexpr weekday   Sunday{0};
// inline constexpr weekday   Monday{1};
// inline constexpr weekday   Tuesday{2};
// inline constexpr weekday   Wednesday{3};
// inline constexpr weekday   Thursday{4};
// inline constexpr weekday   Friday{5};
// inline constexpr weekday   Saturday{6};


#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Sunday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Monday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Tuesday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Wednesday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Thursday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Friday));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::weekday, decltype(cuda_for_dali::std::chrono::Saturday));

    static_assert( cuda_for_dali::std::chrono::Sunday    == cuda_for_dali::std::chrono::weekday(0),  "");
    static_assert( cuda_for_dali::std::chrono::Monday    == cuda_for_dali::std::chrono::weekday(1),  "");
    static_assert( cuda_for_dali::std::chrono::Tuesday   == cuda_for_dali::std::chrono::weekday(2),  "");
    static_assert( cuda_for_dali::std::chrono::Wednesday == cuda_for_dali::std::chrono::weekday(3),  "");
    static_assert( cuda_for_dali::std::chrono::Thursday  == cuda_for_dali::std::chrono::weekday(4),  "");
    static_assert( cuda_for_dali::std::chrono::Friday    == cuda_for_dali::std::chrono::weekday(5),  "");
    static_assert( cuda_for_dali::std::chrono::Saturday  == cuda_for_dali::std::chrono::weekday(6),  "");

    assert(cuda_for_dali::std::chrono::Sunday    == cuda_for_dali::std::chrono::weekday(0));
    assert(cuda_for_dali::std::chrono::Monday    == cuda_for_dali::std::chrono::weekday(1));
    assert(cuda_for_dali::std::chrono::Tuesday   == cuda_for_dali::std::chrono::weekday(2));
    assert(cuda_for_dali::std::chrono::Wednesday == cuda_for_dali::std::chrono::weekday(3));
    assert(cuda_for_dali::std::chrono::Thursday  == cuda_for_dali::std::chrono::weekday(4));
    assert(cuda_for_dali::std::chrono::Friday    == cuda_for_dali::std::chrono::weekday(5));
    assert(cuda_for_dali::std::chrono::Saturday  == cuda_for_dali::std::chrono::weekday(6));

    assert(cuda_for_dali::std::chrono::Sunday.c_encoding()    ==  0);
    assert(cuda_for_dali::std::chrono::Monday.c_encoding()    ==  1);
    assert(cuda_for_dali::std::chrono::Tuesday.c_encoding()   ==  2);
    assert(cuda_for_dali::std::chrono::Wednesday.c_encoding() ==  3);
    assert(cuda_for_dali::std::chrono::Thursday.c_encoding()  ==  4);
    assert(cuda_for_dali::std::chrono::Friday.c_encoding()    ==  5);
    assert(cuda_for_dali::std::chrono::Saturday.c_encoding()  ==  6);

  return 0;
}
