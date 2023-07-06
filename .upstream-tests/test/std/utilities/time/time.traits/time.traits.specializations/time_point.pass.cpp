//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// template <class Clock, class Duration1, class Duration2>
// struct common_type<chrono::time_point<Clock, Duration1>, chrono::time_point<Clock, Duration2>>
// {
//     typedef chrono::time_point<Clock, typename common_type<Duration1, Duration2>::type> type;
// };

#include <cuda_for_dali/std/chrono>

template <class D1, class D2, class De>
__host__ __device__
void
test()
{
    typedef cuda_for_dali::std::chrono::system_clock C;
    typedef cuda_for_dali::std::chrono::time_point<C, D1> T1;
    typedef cuda_for_dali::std::chrono::time_point<C, D2> T2;
    typedef cuda_for_dali::std::chrono::time_point<C, De> Te;
    typedef typename cuda_for_dali::std::common_type<T1, T2>::type Tc;
    static_assert((cuda_for_dali::std::is_same<Tc, Te>::value), "");
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
