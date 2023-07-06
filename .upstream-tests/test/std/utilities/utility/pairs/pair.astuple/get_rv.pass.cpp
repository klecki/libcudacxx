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

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, cuda_for_dali::std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <cuda_for_dali/std/utility>
// cuda/std/memory not supported
// #include <cuda_for_dali/std/memory>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // cuda/std/memory not supported
    /*
    {
        typedef cuda_for_dali::std::pair<cuda_for_dali::std::unique_ptr<int>, short> P;
        P p(cuda_for_dali::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
        cuda_for_dali::std::unique_ptr<int> ptr = cuda_for_dali::std::get<0>(cuda_for_dali::std::move(p));
        assert(*ptr == 3);
    }
    */
  return 0;
}
