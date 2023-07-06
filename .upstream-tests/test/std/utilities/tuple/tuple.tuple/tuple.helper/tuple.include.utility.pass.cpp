//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };
//
//  LWG #2212 says that tuple_size and tuple_element must be
//     available after including <utility>

#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/utility>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"

template <class T, cuda_for_dali::std::size_t N, class U, size_t idx>
__host__ __device__ void test()
{
    static_assert((cuda_for_dali::std::is_base_of<cuda_for_dali::std::integral_constant<cuda_for_dali::std::size_t, N>,
                                   cuda_for_dali::std::tuple_size<T> >::value), "");
    static_assert((cuda_for_dali::std::is_base_of<cuda_for_dali::std::integral_constant<cuda_for_dali::std::size_t, N>,
                                   cuda_for_dali::std::tuple_size<const T> >::value), "");
    static_assert((cuda_for_dali::std::is_base_of<cuda_for_dali::std::integral_constant<cuda_for_dali::std::size_t, N>,
                                   cuda_for_dali::std::tuple_size<volatile T> >::value), "");
    static_assert((cuda_for_dali::std::is_base_of<cuda_for_dali::std::integral_constant<cuda_for_dali::std::size_t, N>,
                                   cuda_for_dali::std::tuple_size<const volatile T> >::value), "");
    static_assert((cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<idx, T>::type, U>::value), "");
    static_assert((cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<idx, const T>::type, const U>::value), "");
    static_assert((cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<idx, volatile T>::type, volatile U>::value), "");
    static_assert((cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<idx, const volatile T>::type, const volatile U>::value), "");
}

int main(int, char**)
{
    test<cuda_for_dali::std::pair<int, int>, 2, int, 0>();
    test<cuda_for_dali::std::pair<int, int>, 2, int, 1>();
    test<cuda_for_dali::std::pair<const int, int>, 2, int, 1>();
    test<cuda_for_dali::std::pair<int, volatile int>, 2, volatile int, 1>();
    test<cuda_for_dali::std::pair<char *, int>, 2, char *, 0>();
    test<cuda_for_dali::std::pair<char *, int>, 2, int,    1>();

  return 0;
}
