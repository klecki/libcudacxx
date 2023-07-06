//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   typename tuple_element<I, tuple<Types...> >::type&
//   get(tuple<Types...>& t);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
// cuda_for_dali::std::string not supported
//#include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 11

struct Empty {};

struct S {
   cuda_for_dali::std::tuple<int, Empty> a;
   int k;
   Empty e;
   __host__ __device__ constexpr S() : a{1,Empty{}}, k(cuda_for_dali::std::get<0>(a)), e(cuda_for_dali::std::get<1>(a)) {}
   };

__host__ __device__ constexpr cuda_for_dali::std::tuple<int, int> getP () { return { 3, 4 }; }
#endif

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::tuple<int> T;
        T t(3);
        assert(cuda_for_dali::std::get<0>(t) == 3);
        cuda_for_dali::std::get<0>(t) = 2;
        assert(cuda_for_dali::std::get<0>(t) == 2);
    }
    // cuda_for_dali::std::string not supported
    /*
    {
        typedef cuda_for_dali::std::tuple<cuda_for_dali::std::string, int> T;
        T t("high", 5);
        assert(cuda_for_dali::std::get<0>(t) == "high");
        assert(cuda_for_dali::std::get<1>(t) == 5);
        cuda_for_dali::std::get<0>(t) = "four";
        cuda_for_dali::std::get<1>(t) = 4;
        assert(cuda_for_dali::std::get<0>(t) == "four");
        assert(cuda_for_dali::std::get<1>(t) == 4);
    }
    {
        typedef cuda_for_dali::std::tuple<double&, cuda_for_dali::std::string, int> T;
        double d = 1.5;
        T t(d, "high", 5);
        assert(cuda_for_dali::std::get<0>(t) == 1.5);
        assert(cuda_for_dali::std::get<1>(t) == "high");
        assert(cuda_for_dali::std::get<2>(t) == 5);
        cuda_for_dali::std::get<0>(t) = 2.5;
        cuda_for_dali::std::get<1>(t) = "four";
        cuda_for_dali::std::get<2>(t) = 4;
        assert(cuda_for_dali::std::get<0>(t) == 2.5);
        assert(cuda_for_dali::std::get<1>(t) == "four");
        assert(cuda_for_dali::std::get<2>(t) == 4);
        assert(d == 2.5);
    }
    */
#if TEST_STD_VER > 11
    { // get on an rvalue tuple
        static_assert ( cuda_for_dali::std::get<0> ( cuda_for_dali::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 0, "" );
        static_assert ( cuda_for_dali::std::get<1> ( cuda_for_dali::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 1, "" );
        static_assert ( cuda_for_dali::std::get<2> ( cuda_for_dali::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 2, "" );
        static_assert ( cuda_for_dali::std::get<3> ( cuda_for_dali::std::make_tuple ( 0.0f, 1, 2.0, 3L )) == 3, "" );
        static_assert(S().k == 1, "");
        static_assert(cuda_for_dali::std::get<1>(getP()) == 4, "");
    }
#endif


  return 0;
}
