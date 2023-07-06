//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

// template<> class complex<float>
// {
// public:
//     explicit constexpr complex(const complex<double>&);
// };

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    const cuda_for_dali::std::complex<double> cd(2.5, 3.5);
    cuda_for_dali::std::complex<float> cf(cd);
    assert(cf.real() == cd.real());
    assert(cf.imag() == cd.imag());
    }
#if TEST_STD_VER >= 11
    {
    constexpr cuda_for_dali::std::complex<double> cd(2.5, 3.5);
    constexpr cuda_for_dali::std::complex<float> cf(cd);
    static_assert(cf.real() == cd.real(), "");
    static_assert(cf.imag() == cd.imag(), "");
    }
#endif

  return 0;
}
