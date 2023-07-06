//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// minus

#include <cuda_for_dali/std/functional>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda_for_dali::std::minus<int> F;
    const F f = F();
    static_assert((cuda_for_dali::std::is_same<int, F::first_argument_type>::value), "" );
    static_assert((cuda_for_dali::std::is_same<int, F::second_argument_type>::value), "" );
    static_assert((cuda_for_dali::std::is_same<int, F::result_type>::value), "" );
    assert(f(3, 2) == 1);
#if TEST_STD_VER > 11
    typedef cuda_for_dali::std::minus<> F2;
    const F2 f2 = F2();
    assert(f2(3,2) == 1);
    assert(f2(3.0, 2) == 1);
    assert(f2(3, 2.5) == 0.5);

    constexpr int foo = cuda_for_dali::std::minus<int> () (3, 2);
    static_assert ( foo == 1, "" );

    constexpr double bar = cuda_for_dali::std::minus<> () (3.0, 2);
    static_assert ( bar == 1.0, "" );
#endif

  return 0;
}
