//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

__host__ __device__
constexpr bool test_ignore_constexpr()
{
// NVCC does not support constexpr non-integral types
#if TEST_STD_VER > 11 && !defined(__CUDA_ARCH__)
    { // Test that std::ignore provides constexpr converting assignment.
        auto& res = (cuda_for_dali::std::ignore = 42);
        assert(&res == &cuda_for_dali::std::ignore);
    }
    { // Test that cuda_for_dali::std::ignore provides constexpr copy/move constructors
        auto copy = cuda_for_dali::std::ignore;
        auto moved = cuda_for_dali::std::move(copy);
        unused(moved);
    }
    { // Test that cuda_for_dali::std::ignore provides constexpr copy/move assignment
        auto copy = cuda_for_dali::std::ignore;
        copy = cuda_for_dali::std::ignore;
        auto moved = cuda_for_dali::std::ignore;
        moved = cuda_for_dali::std::move(copy);
        unused(moved);
    }
#endif
    return true;
}

int main(int, char**) {
    {
        constexpr auto& ignore_v = cuda_for_dali::std::ignore;
        ((void)ignore_v);
    }
    {
        static_assert(test_ignore_constexpr(), "");
    }
    {
        LIBCPP_STATIC_ASSERT(cuda_for_dali::std::is_trivial<decltype(cuda_for_dali::std::ignore)>::value, "");
    }

  return 0;
}
