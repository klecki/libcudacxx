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
//   proj(const complex<T>& x);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& z, cuda_for_dali::std::complex<T> x)
{
    assert(proj(z) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda_for_dali::std::complex<T>(1, 2), cuda_for_dali::std::complex<T>(1, 2));
    test(cuda_for_dali::std::complex<T>(-1, 2), cuda_for_dali::std::complex<T>(-1, 2));
    test(cuda_for_dali::std::complex<T>(1, -2), cuda_for_dali::std::complex<T>(1, -2));
    test(cuda_for_dali::std::complex<T>(-1, -2), cuda_for_dali::std::complex<T>(-1, -2));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda_for_dali::std::complex<double> r = proj(testcases[i]);
        switch (classify(testcases[i]))
        {
        case zero:
        case non_zero:
            assert(r == testcases[i]);
            assert(cuda_for_dali::std::signbit(real(r)) == cuda_for_dali::std::signbit(real(testcases[i])));
            assert(cuda_for_dali::std::signbit(imag(r)) == cuda_for_dali::std::signbit(imag(testcases[i])));
            break;
        case inf:
            assert(cuda_for_dali::std::isinf(real(r)) && real(r) > 0);
            assert(imag(r) == 0);
            assert(cuda_for_dali::std::signbit(imag(r)) == cuda_for_dali::std::signbit(imag(testcases[i])));
            break;
        case NaN:
        case non_zero_nan:
            assert(classify(r) == classify(testcases[i]));
            break;
        }
    }
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();();
    test_edges();

  return 0;
}
