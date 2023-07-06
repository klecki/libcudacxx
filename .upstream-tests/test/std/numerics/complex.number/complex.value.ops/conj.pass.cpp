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
//   conj(const complex<T>& x);

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <class T>
__host__ __device__ void
test(const cuda_for_dali::std::complex<T>& z, cuda_for_dali::std::complex<T> x)
{
    assert(conj(z) == x);
}

template <class T>
__host__ __device__ void
test()
{
    test(cuda_for_dali::std::complex<T>(1, 2), cuda_for_dali::std::complex<T>(1, -2));
    test(cuda_for_dali::std::complex<T>(-1, 2), cuda_for_dali::std::complex<T>(-1, -2));
    test(cuda_for_dali::std::complex<T>(1, -2), cuda_for_dali::std::complex<T>(1, 2));
    test(cuda_for_dali::std::complex<T>(-1, -2), cuda_for_dali::std::complex<T>(-1, 2));
}

int main(int, char**)
{
    test<float>();
    test<double>();
// CUDA treats long double as double
//  test<long double>();();

  return 0;
}
