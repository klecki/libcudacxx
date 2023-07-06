//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// ~pair()

// C++17 added:
//   The destructor of pair shall be a trivial destructor
//     if (is_trivially_destructible_v<T1> && is_trivially_destructible_v<T2>) is true.


#include <cuda_for_dali/std/utility>
#include <cuda_for_dali/std/type_traits>
// cuda_for_dali::std::string not supported
// #include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "DefaultOnly.h"

int main(int, char**)
{
  static_assert((cuda_for_dali::std::is_trivially_destructible<
      cuda_for_dali::std::pair<int, float> >::value), "");
  /*
  static_assert((!cuda_for_dali::std::is_trivially_destructible<
      cuda_for_dali::std::pair<int, cuda_for_dali::std::string> >::value), "");
  */
  static_assert((!cuda_for_dali::std::is_trivially_destructible<
      cuda_for_dali::std::pair<int, DefaultOnly> >::value), "");

  return 0;
}
