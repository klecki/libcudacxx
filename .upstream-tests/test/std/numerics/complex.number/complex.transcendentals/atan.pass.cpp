//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   atan(const complex<T>& x);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& c, cuda_for_dali::std::complex<T> x)
{
    assert(atan(c) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda_for_dali::std::complex<T>(0, 0), cuda_for_dali::std::complex<T>(0, 0));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda_for_dali::std::complex<double> r = atan(testcases[i]);
        cuda_for_dali::std::complex<double> t1(-imag(testcases[i]), real(testcases[i]));
        cuda_for_dali::std::complex<double> t2 = atanh(t1);
        cuda_for_dali::std::complex<double> z(imag(t2), -real(t2));
        if (cuda_for_dali::std::isnan(real(r)))
            assert(cuda_for_dali::std::isnan(real(z)));
        else
        {
            assert(real(r) == real(z));
            assert(cuda_for_dali::std::signbit(real(r)) == cuda_for_dali::std::signbit(real(z)));
        }
        if (cuda_for_dali::std::isnan(imag(r)))
            assert(cuda_for_dali::std::isnan(imag(z)));
        else
        {
            assert(imag(r) == imag(z));
            assert(cuda_for_dali::std::signbit(imag(r)) == cuda_for_dali::std::signbit(imag(z)));
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();
    test_edges();

  return 0;
}
