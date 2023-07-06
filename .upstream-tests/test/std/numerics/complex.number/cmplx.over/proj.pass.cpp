//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>    complex<T>           proj(const complex<T>&);
//                      complex<long double> proj(long double);
//                      complex<double>      proj(double);
// template<Integral T> complex<double>      proj(T);
//                      complex<float>       proj(float);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

template <class T>
__host__ __device__ void
test(T x, typename cuda_for_dali::std::enable_if<cuda_for_dali::std::is_integral<T>::value>::type* = 0)
{
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::proj(x)), cuda_for_dali::std::complex<double> >::value), "");
    assert(cuda_for_dali::std::proj(x) == proj(cuda_for_dali::std::complex<double>(x, 0)));
}

template <class T>
__host__ __device__ void
test(T x, typename cuda_for_dali::std::enable_if<cuda_for_dali::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::proj(x)), cuda_for_dali::std::complex<T> >::value), "");
    assert(cuda_for_dali::std::proj(x) == proj(cuda_for_dali::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void
test(T x, typename cuda_for_dali::std::enable_if<!cuda_for_dali::std::is_integral<T>::value &&
                                  !cuda_for_dali::std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::proj(x)), cuda_for_dali::std::complex<T> >::value), "");
    assert(cuda_for_dali::std::proj(x) == proj(cuda_for_dali::std::complex<T>(x, 0)));
}

template <class T>
__host__ __device__ void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();

  return 0;
}
