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
//   typename tuple_element<I, tuple<Types...> >::type&&
//   get(tuple<Types...>&& t);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::unique_ptr not supported
//#include <cuda_for_dali/std/memory>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"

int main(int, char**)
{
    // cuda_for_dali::std::unique_ptr not supported
    /*
    {
        typedef cuda_for_dali::std::tuple<cuda_for_dali::std::unique_ptr<int> > T;
        T t(cuda_for_dali::std::unique_ptr<int>(new int(3)));
        cuda_for_dali::std::unique_ptr<int> p = cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t));
        assert(*p == 3);
    }
    */
    {
        cuda_for_dali::std::tuple<MoveOnly> t(3);
        MoveOnly _m = cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t));
        assert(_m.get() == 3);
    }
  return 0;
}
