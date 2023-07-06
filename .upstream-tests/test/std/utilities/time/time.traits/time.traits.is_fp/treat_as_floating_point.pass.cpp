//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// treat_as_floating_point

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__
void
test()
{
    static_assert((cuda_for_dali::std::is_base_of<cuda_for_dali::std::is_floating_point<T>,
                                   cuda_for_dali::std::chrono::treat_as_floating_point<T> >::value), "");
#if TEST_STD_VER > 14
    static_assert(cuda_for_dali::std::is_floating_point<T>::value ==
                                  cuda_for_dali::std::chrono::treat_as_floating_point_v<T>, "");
#endif
}

struct A {};

int main(int, char**)
{
    test<int>();
    test<unsigned>();
    test<char>();
    test<bool>();
    test<float>();
    test<double>();
    test<long double>();
    test<A>();

  return 0;
}
