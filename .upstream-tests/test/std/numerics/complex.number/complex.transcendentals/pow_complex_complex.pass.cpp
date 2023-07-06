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
//   pow(const complex<T>& x, const complex<T>& y);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& a, const cuda_for_dali::std::complex<T>& b, cuda_for_dali::std::complex<T> x)
{
    cuda_for_dali::std::complex<T> c = pow(a, b);
    is_about(real(c), real(x));
    is_about(imag(c), imag(x));
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda_for_dali::std::complex<T>(2, 3), cuda_for_dali::std::complex<T>(2, 0), cuda_for_dali::std::complex<T>(-5, 12));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        for (unsigned j = 0; j < N; ++j)
        {
            cuda_for_dali::std::complex<double> r = pow(testcases[i], testcases[j]);
            cuda_for_dali::std::complex<double> z = exp(testcases[j] * log(testcases[i]));
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
