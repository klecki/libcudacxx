//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template<class... Types>
//     tuple<Types&&...> forward_as_tuple(Types&&... t);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <class Tuple>
__host__ __device__ void test0(const Tuple&)
{
    static_assert(cuda_for_dali::std::tuple_size<Tuple>::value == 0, "");
}

template <class Tuple>
__host__ __device__ void test1a(const Tuple& t)
{
    static_assert(cuda_for_dali::std::tuple_size<Tuple>::value == 1, "");
    static_assert(cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<0, Tuple>::type, int&&>::value, "");
    assert(cuda_for_dali::std::get<0>(t) == 1);
}

template <class Tuple>
__host__ __device__ void test1b(const Tuple& t)
{
    static_assert(cuda_for_dali::std::tuple_size<Tuple>::value == 1, "");
    static_assert(cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<0, Tuple>::type, int&>::value, "");
    assert(cuda_for_dali::std::get<0>(t) == 2);
}

template <class Tuple>
__host__ __device__ void test2a(const Tuple& t)
{
    static_assert(cuda_for_dali::std::tuple_size<Tuple>::value == 2, "");
    static_assert(cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<0, Tuple>::type, double&>::value, "");
    static_assert(cuda_for_dali::std::is_same<typename cuda_for_dali::std::tuple_element<1, Tuple>::type, char&>::value, "");
    assert(cuda_for_dali::std::get<0>(t) == 2.5);
    assert(cuda_for_dali::std::get<1>(t) == 'a');
}

#if TEST_STD_VER > 11
template <class Tuple>
__host__ __device__ constexpr int test3(const Tuple&)
{
    return cuda_for_dali::std::tuple_size<Tuple>::value;
}
#endif

int main(int, char**)
{
    {
        test0(cuda_for_dali::std::forward_as_tuple());
    }
    {
        test1a(cuda_for_dali::std::forward_as_tuple(1));
    }
    {
        int i = 2;
        test1b(cuda_for_dali::std::forward_as_tuple(i));
    }
    {
        double i = 2.5;
        char c = 'a';
        test2a(cuda_for_dali::std::forward_as_tuple(i, c));
#if TEST_STD_VER > 11
        static_assert ( test3 (cuda_for_dali::std::forward_as_tuple(i, c)) == 2, "" );
#endif
    }

  return 0;
}
