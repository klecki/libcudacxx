//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_for_dali/std/cstddef>
#include <test_macros.h>

// UNSUPPORTED: c++98, c++03, c++11, c++14

// constexpr byte operator|(byte l, byte r) noexcept;

int main(int, char**) {
    constexpr cuda_for_dali::std::byte b1{static_cast<cuda_for_dali::std::byte>(1)};
    constexpr cuda_for_dali::std::byte b2{static_cast<cuda_for_dali::std::byte>(2)};
    constexpr cuda_for_dali::std::byte b8{static_cast<cuda_for_dali::std::byte>(8)};

    static_assert(noexcept(b1 | b2), "" );

    static_assert(cuda_for_dali::std::to_integer<int>(b1 | b2) ==  3, "");
    static_assert(cuda_for_dali::std::to_integer<int>(b1 | b8) ==  9, "");
    static_assert(cuda_for_dali::std::to_integer<int>(b2 | b8) == 10, "");

    static_assert(cuda_for_dali::std::to_integer<int>(b2 | b1) ==  3, "");
    static_assert(cuda_for_dali::std::to_integer<int>(b8 | b1) ==  9, "");
    static_assert(cuda_for_dali::std::to_integer<int>(b8 | b2) == 10, "");

  return 0;
}
