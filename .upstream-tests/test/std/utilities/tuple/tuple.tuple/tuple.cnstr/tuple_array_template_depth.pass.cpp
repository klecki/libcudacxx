//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Tuple, __tuple_convertible<Tuple, tuple> >
//   tuple(Tuple &&);
//
// template <class Tuple, __tuple_constructible<Tuple, tuple> >
//   tuple(Tuple &&);

// This test checks that we do not evaluate __make_tuple_types
// on the array.

// cuda_for_dali::std::array not supported
// #include <cuda_for_dali/std/array>
#include <cuda_for_dali/std/tuple>

#include "test_macros.h"

// cuda_for_dali::std::array not supported
/*
// Use 1256 to try and blow the template instantiation depth for all compilers.
typedef cuda_for_dali::std::array<char, 1256> array_t;
typedef cuda_for_dali::std::tuple<array_t> tuple_t;
*/

int main(int, char**)
{
  /*
  array_t arr;
  tuple_t tup(arr);
  */
  return 0;
}
