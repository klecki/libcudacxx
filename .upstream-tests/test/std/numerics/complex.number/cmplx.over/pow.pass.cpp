//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const T& x, const complex<U>& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const U& y);

// template<Arithmetic T, Arithmetic U>
//   complex<promote<T, U>::type>
//   pow(const complex<T>& x, const complex<U>& y);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

template <class T>
__host__ __device__ double
promote(T, typename cuda_for_dali::std::enable_if<cuda_for_dali::std::is_integral<T>::value>::type* = 0);

__host__ __device__ float promote(float);
__host__ __device__ double promote(double);
__host__ __device__ long double promote(long double);

template <class T, class U>
__host__ __device__ void
test(T x, const cuda_for_dali::std::complex<U>& y)
{
    typedef decltype(promote(x)+promote(real(y))) V;
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::pow(x, y)), cuda_for_dali::std::complex<V> >::value), "");
    assert(cuda_for_dali::std::pow(x, y) == pow(cuda_for_dali::std::complex<V>(x, 0), cuda_for_dali::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& x, U y)
{
    typedef decltype(promote(real(x))+promote(y)) V;
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::pow(x, y)), cuda_for_dali::std::complex<V> >::value), "");
    assert(cuda_for_dali::std::pow(x, y) == pow(cuda_for_dali::std::complex<V>(x), cuda_for_dali::std::complex<V>(y, 0)));
}

template <class T, class U>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& x, const cuda_for_dali::std::complex<U>& y)
{
    typedef decltype(promote(real(x))+promote(real(y))) V;
    static_assert((cuda_for_dali::std::is_same<decltype(cuda_for_dali::std::pow(x, y)), cuda_for_dali::std::complex<V> >::value), "");
    assert(cuda_for_dali::std::pow(x, y) == pow(cuda_for_dali::std::complex<V>(x), cuda_for_dali::std::complex<V>(y)));
}

template <class T, class U>
__host__ __device__ void
test(typename cuda_for_dali::std::enable_if<cuda_for_dali::std::is_integral<T>::value>::type* = 0, typename cuda_for_dali::std::enable_if<!cuda_for_dali::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), cuda_for_dali::std::complex<U>(4, 5));
    test(cuda_for_dali::std::complex<U>(3, 4), T(5));
}

template <class T, class U>
__host__ __device__ void
test(typename cuda_for_dali::std::enable_if<!cuda_for_dali::std::is_integral<T>::value>::type* = 0, typename cuda_for_dali::std::enable_if<!cuda_for_dali::std::is_integral<U>::value>::type* = 0)
{
    test(T(3), cuda_for_dali::std::complex<U>(4, 5));
    test(cuda_for_dali::std::complex<T>(3, 4), U(5));
    test(cuda_for_dali::std::complex<T>(3, 4), cuda_for_dali::std::complex<U>(5, 6));
}

int main(int, char**)
{
    test<int, float>();
    test<int, double>();

    test<unsigned, float>();
    test<unsigned, double>();

    test<long long, float>();
    test<long long, double>();

    test<float, double>();

    test<double, float>();

// CUDA treats long double as double
//  test<int, long double>();
//  test<unsigned, long double>();
//  test<long long, long double>();
//  test<float, long double>();
//  test<double, long double>();
//  test<long double, float>();
//  test<long double, double>();

  return 0;
}
