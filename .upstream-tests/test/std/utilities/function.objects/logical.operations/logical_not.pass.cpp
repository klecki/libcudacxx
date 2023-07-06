//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// logical_not

#include <cuda_for_dali/std/functional>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda_for_dali::std::logical_not<int> F;
    const F f = F();
    static_assert((cuda_for_dali::std::is_same<F::argument_type, int>::value), "" );
    static_assert((cuda_for_dali::std::is_same<F::result_type, bool>::value), "" );
    assert(!f(36));
    assert(f(0));
#if TEST_STD_VER > 11
    typedef cuda_for_dali::std::logical_not<> F2;
    const F2 f2 = F2();
    assert(!f2(36));
    assert( f2(0));
    assert(!f2(36L));
    assert( f2(0L));

    constexpr bool foo = cuda_for_dali::std::logical_not<int> () (36);
    static_assert ( !foo, "" );

    constexpr bool bar = cuda_for_dali::std::logical_not<> () (36);
    static_assert ( !bar, "" );
#endif

  return 0;
}
