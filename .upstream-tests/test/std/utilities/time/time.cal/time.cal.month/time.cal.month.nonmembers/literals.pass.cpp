//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11

// <chrono>

// inline constexpr month January{1};
// inline constexpr month February{2};
// inline constexpr month March{3};
// inline constexpr month April{4};
// inline constexpr month May{5};
// inline constexpr month June{6};
// inline constexpr month July{7};
// inline constexpr month August{8};
// inline constexpr month September{9};
// inline constexpr month October{10};
// inline constexpr month November{11};
// inline constexpr month December{12};


#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{

    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::January));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::February));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::March));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::April));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::May));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::June));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::July));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::August));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::September));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::October));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::November));
    ASSERT_SAME_TYPE(const cuda_for_dali::std::chrono::month, decltype(cuda_for_dali::std::chrono::December));

    static_assert( cuda_for_dali::std::chrono::January   == cuda_for_dali::std::chrono::month(1),  "");
    static_assert( cuda_for_dali::std::chrono::February  == cuda_for_dali::std::chrono::month(2),  "");
    static_assert( cuda_for_dali::std::chrono::March     == cuda_for_dali::std::chrono::month(3),  "");
    static_assert( cuda_for_dali::std::chrono::April     == cuda_for_dali::std::chrono::month(4),  "");
    static_assert( cuda_for_dali::std::chrono::May       == cuda_for_dali::std::chrono::month(5),  "");
    static_assert( cuda_for_dali::std::chrono::June      == cuda_for_dali::std::chrono::month(6),  "");
    static_assert( cuda_for_dali::std::chrono::July      == cuda_for_dali::std::chrono::month(7),  "");
    static_assert( cuda_for_dali::std::chrono::August    == cuda_for_dali::std::chrono::month(8),  "");
    static_assert( cuda_for_dali::std::chrono::September == cuda_for_dali::std::chrono::month(9),  "");
    static_assert( cuda_for_dali::std::chrono::October   == cuda_for_dali::std::chrono::month(10), "");
    static_assert( cuda_for_dali::std::chrono::November  == cuda_for_dali::std::chrono::month(11), "");
    static_assert( cuda_for_dali::std::chrono::December  == cuda_for_dali::std::chrono::month(12), "");

    assert(cuda_for_dali::std::chrono::January   == cuda_for_dali::std::chrono::month(1));
    assert(cuda_for_dali::std::chrono::February  == cuda_for_dali::std::chrono::month(2));
    assert(cuda_for_dali::std::chrono::March     == cuda_for_dali::std::chrono::month(3));
    assert(cuda_for_dali::std::chrono::April     == cuda_for_dali::std::chrono::month(4));
    assert(cuda_for_dali::std::chrono::May       == cuda_for_dali::std::chrono::month(5));
    assert(cuda_for_dali::std::chrono::June      == cuda_for_dali::std::chrono::month(6));
    assert(cuda_for_dali::std::chrono::July      == cuda_for_dali::std::chrono::month(7));
    assert(cuda_for_dali::std::chrono::August    == cuda_for_dali::std::chrono::month(8));
    assert(cuda_for_dali::std::chrono::September == cuda_for_dali::std::chrono::month(9));
    assert(cuda_for_dali::std::chrono::October   == cuda_for_dali::std::chrono::month(10));
    assert(cuda_for_dali::std::chrono::November  == cuda_for_dali::std::chrono::month(11));
    assert(cuda_for_dali::std::chrono::December  == cuda_for_dali::std::chrono::month(12));

    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::January)   ==  1);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::February)  ==  2);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::March)     ==  3);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::April)     ==  4);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::May)       ==  5);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::June)      ==  6);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::July)      ==  7);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::August)    ==  8);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::September) ==  9);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::October)   == 10);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::November)  == 11);
    assert(static_cast<unsigned>(cuda_for_dali::std::chrono::December)  == 12);

  return 0;
}
