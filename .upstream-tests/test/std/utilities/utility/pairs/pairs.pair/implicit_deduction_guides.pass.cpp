//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: nvcc-10.3, nvcc-11.0, nvcc-11.1, nvcc-11.2, nvcc-11.3, nvcc-11.4

// GCC's implementation of class template deduction is still immature and runs
// into issues with libc++. However GCC accepts this code when compiling
// against libstdc++.
// XFAIL: gcc

// Currently broken with Clang + NVCC.
// XFAIL: clang-6, clang-7, clang-9, clang-10

// <utility>

// Test that the constructors offered by cuda_for_dali::std::pair are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <cuda_for_dali/std/utility>
// cuda/std/memory not supported
// #include <cuda_for_dali/std/memory>
// cuda_for_dali::std::string not supported
// #include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

// Overloads
// ---------------
// (1)  pair(const T1&, const T2&) -> pair<T1, T2>
// (2)  explicit pair(const T1&, const T2&) -> pair<T1, T2>
// (3)  pair(pair const& t) -> decltype(t)
// (4)  pair(pair&& t) -> decltype(t)
// (5)  pair(pair<U1, U2> const&) -> pair<U1, U2>
// (6)  explicit pair(pair<U1, U2> const&) -> pair<U1, U2>
// (7)  pair(pair<U1, U2> &&) -> pair<U1, U2>
// (8)  explicit pair(pair<U1, U2> &&) -> pair<U1, U2>
int main(int, char**)
{
  using E = ExplicitTestTypes::TestType;
  static_assert(!cuda_for_dali::std::is_convertible<E const&, E>::value, "");
  { // Testing (1)
    int const x = 42;
    cuda_for_dali::std::pair t1("abc", x);
    ASSERT_SAME_TYPE(decltype(t1), cuda_for_dali::std::pair<const char*, int>);
    unused(t1);
  }
  { // Testing (2)
    cuda_for_dali::std::pair p1(E{}, 42);
    ASSERT_SAME_TYPE(decltype(p1), cuda_for_dali::std::pair<E, int>);
    unused(p1);

    const E t{};
    cuda_for_dali::std::pair p2(t, E{});
    ASSERT_SAME_TYPE(decltype(p2), cuda_for_dali::std::pair<E, E>);
  }
  { // Testing (3, 5)
    cuda_for_dali::std::pair<double, decltype(nullptr)> const p(0.0, nullptr);
    cuda_for_dali::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), cuda_for_dali::std::pair<double, decltype(nullptr)>);
  }
  { // Testing (3, 6)
    cuda_for_dali::std::pair<E, decltype(nullptr)> const p(E{}, nullptr);
    cuda_for_dali::std::pair p1(p);
    unused(p1);
    ASSERT_SAME_TYPE(decltype(p1), cuda_for_dali::std::pair<E, decltype(nullptr)>);
  }
  // cuda_for_dali::std::string not supported
  /*
  { // Testing (4, 7)
    cuda_for_dali::std::pair<cuda_for_dali::std::string, void*> p("abc", nullptr);
    cuda_for_dali::std::pair p1(cuda_for_dali::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), cuda_for_dali::std::pair<cuda_for_dali::std::string, void*>);
  }
  { // Testing (4, 8)
    cuda_for_dali::std::pair<cuda_for_dali::std::string, E> p("abc", E{});
    cuda_for_dali::std::pair p1(cuda_for_dali::std::move(p));
    ASSERT_SAME_TYPE(decltype(p1), cuda_for_dali::std::pair<cuda_for_dali::std::string, E>);
  }
  */
  return 0;
}
