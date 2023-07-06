//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/type_traits>

// constexpr bool is_constant_evaluated() noexcept; // C++20

#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

// libcudacxx does not have feature test macros... yet
/*
#ifndef __cpp_lib_is_constant_evaluated
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
# error __cpp_lib_is_constant_evaluated should be defined
#endif
#endif
*/

template <bool> struct InTemplate {};

int main(int, char**)
{
#if defined(_LIBCUDAFORDALICXX_IS_CONSTANT_EVALUATED)
  // Test the signature
  {
    ASSERT_SAME_TYPE(decltype(cuda_for_dali::std::is_constant_evaluated()), bool);
    ASSERT_NOEXCEPT(cuda_for_dali::std::is_constant_evaluated());
    constexpr bool p = cuda_for_dali::std::is_constant_evaluated();
    assert(p);
  }
  // Test the return value of the builtin for basic sanity only. It's the
  // compilers job to test tho builtin for correctness.
  {
    static_assert(cuda_for_dali::std::is_constant_evaluated(), "");
    bool p = cuda_for_dali::std::is_constant_evaluated();
    assert(!p);
    ASSERT_SAME_TYPE(InTemplate<cuda_for_dali::std::is_constant_evaluated()>, InTemplate<true>);
    static int local_static = cuda_for_dali::std::is_constant_evaluated() ? 42 : -1;
    assert(local_static == 42);
  }
#endif
  return 0;
}
