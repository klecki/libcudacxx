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
//   sqrt(const complex<T>& x);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "../cases.h"

template <class T>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& c, cuda_for_dali::std::complex<T> x)
{
    cuda_for_dali::std::complex<T> a = sqrt(c);
    is_about(real(a), real(x));
    assert(cuda_for_dali::std::abs(imag(c)) < 1.e-6);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda_for_dali::std::complex<T>(64, 0), cuda_for_dali::std::complex<T>(8, 0));
}

__host__ __device__ void test_edges()
{
    auto testcases = get_testcases();
    const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
    for (unsigned i = 0; i < N; ++i)
    {
        cuda_for_dali::std::complex<double> r = sqrt(testcases[i]);
        if (testcases[i].real() == 0 && testcases[i].imag() == 0)
        {
            assert(!cuda_for_dali::std::signbit(r.real()));
            assert(cuda_for_dali::std::signbit(r.imag()) == cuda_for_dali::std::signbit(testcases[i].imag()));
        }
        else if (cuda_for_dali::std::isinf(testcases[i].imag()))
        {
            assert(cuda_for_dali::std::isinf(r.real()));
            assert(r.real() > 0);
            assert(cuda_for_dali::std::isinf(r.imag()));
            assert(cuda_for_dali::std::signbit(r.imag()) == cuda_for_dali::std::signbit(testcases[i].imag()));
        }
        else if (cuda_for_dali::std::isfinite(testcases[i].real()) && cuda_for_dali::std::isnan(testcases[i].imag()))
        {
            assert(cuda_for_dali::std::isnan(r.real()));
            assert(cuda_for_dali::std::isnan(r.imag()));
        }
        else if (cuda_for_dali::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda_for_dali::std::isfinite(testcases[i].imag()))
        {
            assert(r.real() == 0);
            assert(!cuda_for_dali::std::signbit(r.real()));
            assert(cuda_for_dali::std::isinf(r.imag()));
            assert(cuda_for_dali::std::signbit(testcases[i].imag()) == cuda_for_dali::std::signbit(r.imag()));
        }
        else if (cuda_for_dali::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda_for_dali::std::isfinite(testcases[i].imag()))
        {
            assert(cuda_for_dali::std::isinf(r.real()));
            assert(r.real() > 0);
            assert(r.imag() == 0);
            assert(cuda_for_dali::std::signbit(testcases[i].imag()) == cuda_for_dali::std::signbit(r.imag()));
        }
        else if (cuda_for_dali::std::isinf(testcases[i].real()) && testcases[i].real() < 0 && cuda_for_dali::std::isnan(testcases[i].imag()))
        {
            assert(cuda_for_dali::std::isnan(r.real()));
            assert(cuda_for_dali::std::isinf(r.imag()));
        }
        else if (cuda_for_dali::std::isinf(testcases[i].real()) && testcases[i].real() > 0 && cuda_for_dali::std::isnan(testcases[i].imag()))
        {
            assert(cuda_for_dali::std::isinf(r.real()));
            assert(r.real() > 0);
            assert(cuda_for_dali::std::isnan(r.imag()));
        }
        else if (cuda_for_dali::std::isnan(testcases[i].real()) && (cuda_for_dali::std::isfinite(testcases[i].imag()) || cuda_for_dali::std::isnan(testcases[i].imag())))
        {
            assert(cuda_for_dali::std::isnan(r.real()));
            assert(cuda_for_dali::std::isnan(r.imag()));
        }
        else if (cuda_for_dali::std::signbit(testcases[i].imag()))
        {
            assert(!cuda_for_dali::std::signbit(r.real()));
            assert(cuda_for_dali::std::signbit(r.imag()));
        }
        else
        {
            assert(!cuda_for_dali::std::signbit(r.real()));
            assert(!cuda_for_dali::std::signbit(r.imag()));
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
