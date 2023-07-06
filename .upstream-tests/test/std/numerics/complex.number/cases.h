//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <complex>

// test cases

#ifndef CASES_H
#define CASES_H

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

using testcases_t = cuda_for_dali::std::complex<double>[137];

struct _testcases {
    testcases_t _cases;

    static constexpr size_t count = sizeof(_cases) / sizeof(_cases[0]);

    __host__ __device__  const cuda_for_dali::std::complex<double>* begin() const {
        return  &_cases[0];
    }
    __host__ __device__  const cuda_for_dali::std::complex<double>* cbegin() const {
        return  &_cases[0];
    }
    __host__ __device__  cuda_for_dali::std::complex<double>* begin() {
        return  &_cases[0];
    }

    __host__ __device__  const cuda_for_dali::std::complex<double>* end() const {
        return  &_cases[count];
    }
    __host__ __device__  const cuda_for_dali::std::complex<double>* cend() const {
        return  &_cases[count];
    }
    __host__ __device__  cuda_for_dali::std::complex<double>* end() {
        return  &_cases[count];
    }

    __host__ __device__  cuda_for_dali::std::complex<double>& operator[](size_t n) {
        return _cases[n];
    }

    __host__ __device__  const cuda_for_dali::std::complex<double>& operator[](size_t n) const {
        return _cases[n];
    }
};

__host__ __device__ _testcases get_testcases() {
    _testcases tc {
        cuda_for_dali::std::complex<double>( 1.e-6,  1.e-6),
        cuda_for_dali::std::complex<double>(-1.e-6,  1.e-6),
        cuda_for_dali::std::complex<double>(-1.e-6, -1.e-6),
        cuda_for_dali::std::complex<double>( 1.e-6, -1.e-6),

        cuda_for_dali::std::complex<double>( 1.e+6,  1.e-6),
        cuda_for_dali::std::complex<double>(-1.e+6,  1.e-6),
        cuda_for_dali::std::complex<double>(-1.e+6, -1.e-6),
        cuda_for_dali::std::complex<double>( 1.e+6, -1.e-6),

        cuda_for_dali::std::complex<double>( 1.e-6,  1.e+6),
        cuda_for_dali::std::complex<double>(-1.e-6,  1.e+6),
        cuda_for_dali::std::complex<double>(-1.e-6, -1.e+6),
        cuda_for_dali::std::complex<double>( 1.e-6, -1.e+6),

        cuda_for_dali::std::complex<double>( 1.e+6,  1.e+6),
        cuda_for_dali::std::complex<double>(-1.e+6,  1.e+6),
        cuda_for_dali::std::complex<double>(-1.e+6, -1.e+6),
        cuda_for_dali::std::complex<double>( 1.e+6, -1.e+6),

        cuda_for_dali::std::complex<double>(NAN, NAN),
        cuda_for_dali::std::complex<double>(-INFINITY, NAN),
        cuda_for_dali::std::complex<double>(-2, NAN),
        cuda_for_dali::std::complex<double>(-1, NAN),
        cuda_for_dali::std::complex<double>(-0.5, NAN),
        cuda_for_dali::std::complex<double>(-0., NAN),
        cuda_for_dali::std::complex<double>(+0., NAN),
        cuda_for_dali::std::complex<double>(0.5, NAN),
        cuda_for_dali::std::complex<double>(1, NAN),
        cuda_for_dali::std::complex<double>(2, NAN),
        cuda_for_dali::std::complex<double>(INFINITY, NAN),

        cuda_for_dali::std::complex<double>(NAN, -INFINITY),
        cuda_for_dali::std::complex<double>(-INFINITY, -INFINITY),
        cuda_for_dali::std::complex<double>(-2, -INFINITY),
        cuda_for_dali::std::complex<double>(-1, -INFINITY),
        cuda_for_dali::std::complex<double>(-0.5, -INFINITY),
        cuda_for_dali::std::complex<double>(-0., -INFINITY),
        cuda_for_dali::std::complex<double>(+0., -INFINITY),
        cuda_for_dali::std::complex<double>(0.5, -INFINITY),
        cuda_for_dali::std::complex<double>(1, -INFINITY),
        cuda_for_dali::std::complex<double>(2, -INFINITY),
        cuda_for_dali::std::complex<double>(INFINITY, -INFINITY),

        cuda_for_dali::std::complex<double>(NAN, -2),
        cuda_for_dali::std::complex<double>(-INFINITY, -2),
        cuda_for_dali::std::complex<double>(-2, -2),
        cuda_for_dali::std::complex<double>(-1, -2),
        cuda_for_dali::std::complex<double>(-0.5, -2),
        cuda_for_dali::std::complex<double>(-0., -2),
        cuda_for_dali::std::complex<double>(+0., -2),
        cuda_for_dali::std::complex<double>(0.5, -2),
        cuda_for_dali::std::complex<double>(1, -2),
        cuda_for_dali::std::complex<double>(2, -2),
        cuda_for_dali::std::complex<double>(INFINITY, -2),

        cuda_for_dali::std::complex<double>(NAN, -1),
        cuda_for_dali::std::complex<double>(-INFINITY, -1),
        cuda_for_dali::std::complex<double>(-2, -1),
        cuda_for_dali::std::complex<double>(-1, -1),
        cuda_for_dali::std::complex<double>(-0.5, -1),
        cuda_for_dali::std::complex<double>(-0., -1),
        cuda_for_dali::std::complex<double>(+0., -1),
        cuda_for_dali::std::complex<double>(0.5, -1),
        cuda_for_dali::std::complex<double>(1, -1),
        cuda_for_dali::std::complex<double>(2, -1),
        cuda_for_dali::std::complex<double>(INFINITY, -1),

        cuda_for_dali::std::complex<double>(NAN, -0.5),
        cuda_for_dali::std::complex<double>(-INFINITY, -0.5),
        cuda_for_dali::std::complex<double>(-2, -0.5),
        cuda_for_dali::std::complex<double>(-1, -0.5),
        cuda_for_dali::std::complex<double>(-0.5, -0.5),
        cuda_for_dali::std::complex<double>(-0., -0.5),
        cuda_for_dali::std::complex<double>(+0., -0.5),
        cuda_for_dali::std::complex<double>(0.5, -0.5),
        cuda_for_dali::std::complex<double>(1, -0.5),
        cuda_for_dali::std::complex<double>(2, -0.5),
        cuda_for_dali::std::complex<double>(INFINITY, -0.5),

        cuda_for_dali::std::complex<double>(NAN, -0.),
        cuda_for_dali::std::complex<double>(-INFINITY, -0.),
        cuda_for_dali::std::complex<double>(-2, -0.),
        cuda_for_dali::std::complex<double>(-1, -0.),
        cuda_for_dali::std::complex<double>(-0.5, -0.),
        cuda_for_dali::std::complex<double>(-0., -0.),
        cuda_for_dali::std::complex<double>(+0., -0.),
        cuda_for_dali::std::complex<double>(0.5, -0.),
        cuda_for_dali::std::complex<double>(1, -0.),
        cuda_for_dali::std::complex<double>(2, -0.),
        cuda_for_dali::std::complex<double>(INFINITY, -0.),

        cuda_for_dali::std::complex<double>(NAN, +0.),
        cuda_for_dali::std::complex<double>(-INFINITY, +0.),
        cuda_for_dali::std::complex<double>(-2, +0.),
        cuda_for_dali::std::complex<double>(-1, +0.),
        cuda_for_dali::std::complex<double>(-0.5, +0.),
        cuda_for_dali::std::complex<double>(-0., +0.),
        cuda_for_dali::std::complex<double>(+0., +0.),
        cuda_for_dali::std::complex<double>(0.5, +0.),
        cuda_for_dali::std::complex<double>(1, +0.),
        cuda_for_dali::std::complex<double>(2, +0.),
        cuda_for_dali::std::complex<double>(INFINITY, +0.),

        cuda_for_dali::std::complex<double>(NAN, 0.5),
        cuda_for_dali::std::complex<double>(-INFINITY, 0.5),
        cuda_for_dali::std::complex<double>(-2, 0.5),
        cuda_for_dali::std::complex<double>(-1, 0.5),
        cuda_for_dali::std::complex<double>(-0.5, 0.5),
        cuda_for_dali::std::complex<double>(-0., 0.5),
        cuda_for_dali::std::complex<double>(+0., 0.5),
        cuda_for_dali::std::complex<double>(0.5, 0.5),
        cuda_for_dali::std::complex<double>(1, 0.5),
        cuda_for_dali::std::complex<double>(2, 0.5),
        cuda_for_dali::std::complex<double>(INFINITY, 0.5),

        cuda_for_dali::std::complex<double>(NAN, 1),
        cuda_for_dali::std::complex<double>(-INFINITY, 1),
        cuda_for_dali::std::complex<double>(-2, 1),
        cuda_for_dali::std::complex<double>(-1, 1),
        cuda_for_dali::std::complex<double>(-0.5, 1),
        cuda_for_dali::std::complex<double>(-0., 1),
        cuda_for_dali::std::complex<double>(+0., 1),
        cuda_for_dali::std::complex<double>(0.5, 1),
        cuda_for_dali::std::complex<double>(1, 1),
        cuda_for_dali::std::complex<double>(2, 1),
        cuda_for_dali::std::complex<double>(INFINITY, 1),

        cuda_for_dali::std::complex<double>(NAN, 2),
        cuda_for_dali::std::complex<double>(-INFINITY, 2),
        cuda_for_dali::std::complex<double>(-2, 2),
        cuda_for_dali::std::complex<double>(-1, 2),
        cuda_for_dali::std::complex<double>(-0.5, 2),
        cuda_for_dali::std::complex<double>(-0., 2),
        cuda_for_dali::std::complex<double>(+0., 2),
        cuda_for_dali::std::complex<double>(0.5, 2),
        cuda_for_dali::std::complex<double>(1, 2),
        cuda_for_dali::std::complex<double>(2, 2),
        cuda_for_dali::std::complex<double>(INFINITY, 2),

        cuda_for_dali::std::complex<double>(NAN, INFINITY),
        cuda_for_dali::std::complex<double>(-INFINITY, INFINITY),
        cuda_for_dali::std::complex<double>(-2, INFINITY),
        cuda_for_dali::std::complex<double>(-1, INFINITY),
        cuda_for_dali::std::complex<double>(-0.5, INFINITY),
        cuda_for_dali::std::complex<double>(-0., INFINITY),
        cuda_for_dali::std::complex<double>(+0., INFINITY),
        cuda_for_dali::std::complex<double>(0.5, INFINITY),
        cuda_for_dali::std::complex<double>(1, INFINITY),
        cuda_for_dali::std::complex<double>(2, INFINITY),
        cuda_for_dali::std::complex<double>(INFINITY, INFINITY)
    };

    return tc;
}

enum {zero, non_zero, inf, NaN, non_zero_nan};

template <class T>
__host__ __device__ int
classify(const cuda_for_dali::std::complex<T>& x)
{
    if (x == cuda_for_dali::std::complex<T>())
        return zero;
    if (cuda_for_dali::std::isinf(x.real()) || cuda_for_dali::std::isinf(x.imag()))
        return inf;
    if (cuda_for_dali::std::isnan(x.real()) && cuda_for_dali::std::isnan(x.imag()))
        return NaN;
    if (cuda_for_dali::std::isnan(x.real()))
    {
        if (x.imag() == T(0))
            return NaN;
        return non_zero_nan;
    }
    if (cuda_for_dali::std::isnan(x.imag()))
    {
        if (x.real() == T(0))
            return NaN;
        return non_zero_nan;
    }
    return non_zero;
}

inline
__host__ __device__ int
classify(double x)
{
    if (x == 0)
        return zero;
    if (cuda_for_dali::std::isinf(x))
        return inf;
    if (cuda_for_dali::std::isnan(x))
        return NaN;
    return non_zero;
}

__host__ __device__ void is_about(float x, float y)
{
    assert(cuda_for_dali::std::abs((x-y)/(x+y)) < 1.e-6);
}

__host__ __device__ void is_about(double x, double y)
{
    assert(cuda_for_dali::std::abs((x-y)/(x+y)) < 1.e-14);
}

// CUDA treats long double as double
/*
__host__ __device__ void is_about(long double x, long double y)
{
    assert(cuda_for_dali::std::abs((x-y)/(x+y)) < 1.e-14);
}
*/
#endif  // CASES_H
