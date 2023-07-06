//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// template <class Rep1, class Period1, class Rep2, class Period2>
// struct common_type<chrono::duration<Rep1, Period1>, chrono::duration<Rep2, Period2>>
// {
//     typedef chrono::duration<typename common_type<Rep1, Rep2>::type, see below }> type;
// };

#include <cuda_for_dali/std/chrono>

template <class D1, class D2, class De>
__host__ __device__
void
test()
{
    typedef typename cuda_for_dali::std::common_type<D1, D2>::type Dc;
    static_assert((cuda_for_dali::std::is_same<Dc, De>::value), "");
}

int main(int, char**)
{
    test<cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1, 100> >,
         cuda_for_dali::std::chrono::duration<long, cuda_for_dali::std::ratio<1, 1000> >,
         cuda_for_dali::std::chrono::duration<long, cuda_for_dali::std::ratio<1, 1000> > >();
    test<cuda_for_dali::std::chrono::duration<long, cuda_for_dali::std::ratio<1, 100> >,
         cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1, 1000> >,
         cuda_for_dali::std::chrono::duration<long, cuda_for_dali::std::ratio<1, 1000> > >();
    test<cuda_for_dali::std::chrono::duration<char, cuda_for_dali::std::ratio<1, 30> >,
         cuda_for_dali::std::chrono::duration<short, cuda_for_dali::std::ratio<1, 1000> >,
         cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<1, 3000> > >();
    test<cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<21, 1> >,
         cuda_for_dali::std::chrono::duration<short, cuda_for_dali::std::ratio<15, 1> >,
         cuda_for_dali::std::chrono::duration<double, cuda_for_dali::std::ratio<3, 1> > >();

  return 0;
}
