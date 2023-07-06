//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <cuda/std/tuple>

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::array not supported
//#include <cuda_for_dali/std/array>

#include "test_macros.h"

template <class Tuple, int Expect>
__host__ __device__ void test()
{
    static_assert(cuda_for_dali::std::tuple_size_v<Tuple> == Expect, "");
    static_assert(cuda_for_dali::std::tuple_size_v<Tuple> == cuda_for_dali::std::tuple_size<Tuple>::value, "");
    static_assert(cuda_for_dali::std::tuple_size_v<Tuple const> == cuda_for_dali::std::tuple_size<Tuple>::value, "");
    static_assert(cuda_for_dali::std::tuple_size_v<Tuple volatile> == cuda_for_dali::std::tuple_size<Tuple>::value, "");
    static_assert(cuda_for_dali::std::tuple_size_v<Tuple const volatile> == cuda_for_dali::std::tuple_size<Tuple>::value, "");
}

int main(int, char**)
{
    test<cuda_for_dali::std::tuple<>, 0>();

    test<cuda_for_dali::std::tuple<int>, 1>();
    // cuda_for_dali::std::array not supported
    //test<cuda_for_dali::std::array<int, 1>, 1>();

    test<cuda_for_dali::std::tuple<int, int>, 2>();
    test<cuda_for_dali::std::pair<int, int>, 2>();
    // cuda_for_dali::std::array not supported
    //test<cuda_for_dali::std::array<int, 2>, 2>();

    test<cuda_for_dali::std::tuple<int, int, int>, 3>();
    // cuda_for_dali::std::array not supported
    //test<cuda_for_dali::std::array<int, 3>, 3>();

  return 0;
}
