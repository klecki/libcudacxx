//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(const tuple& u) = default;

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct Empty {};

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::tuple<> T;
        T t0;
        T t = t0;
        unused(t); // Prevent unused warning
    }
    {
        typedef cuda_for_dali::std::tuple<int> T;
        T t0(2);
        T t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
    }
    {
        typedef cuda_for_dali::std::tuple<int, char> T;
        T t0(2, 'a');
        T t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
        assert(cuda_for_dali::std::get<1>(t) == 'a');
    }
    // cuda_for_dali::std::string not supported
    /*
    {
        typedef cuda_for_dali::std::tuple<int, char, cuda_for_dali::std::string> T;
        const T t0(2, 'a', "some text");
        T t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
        assert(cuda_for_dali::std::get<1>(t) == 'a');
        assert(cuda_for_dali::std::get<2>(t) == "some text");
    }
    */
#if TEST_STD_VER > 11
    {
        typedef cuda_for_dali::std::tuple<int> T;
        constexpr T t0(2);
        constexpr T t = t0;
        static_assert(cuda_for_dali::std::get<0>(t) == 2, "");
    }
    {
        typedef cuda_for_dali::std::tuple<Empty> T;
        constexpr T t0;
        constexpr T t = t0;
        constexpr Empty e = cuda_for_dali::std::get<0>(t);
        ((void)e); // Prevent unused warning
    }
#endif

  return 0;
}
