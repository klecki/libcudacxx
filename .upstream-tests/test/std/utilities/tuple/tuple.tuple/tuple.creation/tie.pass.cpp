//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class... Types>
//   tuple<Types&...> tie(Types&... t);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>

// cuda_for_dali::std::string not supported
// #include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER > 11
__host__ __device__ constexpr bool test_tie_constexpr() {
    {
        int i = 42;
        double f = 1.1;
        using ExpectT = cuda_for_dali::std::tuple<int&, decltype(cuda_for_dali::std::ignore)&, double&>;
        auto res = cuda_for_dali::std::tie(i, cuda_for_dali::std::ignore, f);
        static_assert(cuda_for_dali::std::is_same<ExpectT, decltype(res)>::value, "");
        assert(&cuda_for_dali::std::get<0>(res) == &i);
        assert(&cuda_for_dali::std::get<1>(res) == &cuda_for_dali::std::ignore);
        assert(&cuda_for_dali::std::get<2>(res) == &f);
        // FIXME: If/when tuple gets constexpr assignment
        //res = cuda_for_dali::std::make_tuple(101, nullptr, -1.0);
    }
    return true;
}
#endif

int main(int, char**)
{
    {
        int i = 0;
        const char *_s = "C++";
        // cuda_for_dali::std::string not supported
        // cuda_for_dali::std::string s;
        const char *s;
        cuda_for_dali::std::tie(i, cuda_for_dali::std::ignore, s) = cuda_for_dali::std::make_tuple(42, 3.14, _s);
        assert(i == 42);
        assert(s == _s);
    }
#if TEST_STD_VER > 11
    {
        static constexpr int i = 42;
        static constexpr double f = 1.1;
        constexpr cuda_for_dali::std::tuple<const int &, const double &> t = cuda_for_dali::std::tie(i, f);
        static_assert ( cuda_for_dali::std::get<0>(t) == 42, "" );
        static_assert ( cuda_for_dali::std::get<1>(t) == 1.1, "" );
    }
    {
        static_assert(test_tie_constexpr(), "");
    }
#endif

  return 0;
}
