//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: nvrtc

#include <cuda_for_dali/std/utility>
#include <cuda_for_dali/std/complex>

#include <cuda_for_dali/std/cassert>

int main(int, char**)
{
    typedef cuda_for_dali::std::complex<float> cf;
    auto t1 = cuda_for_dali::std::make_pair<int, int> ( 42, 43 );
    assert ( cuda_for_dali::std::get<int>(t1) == 42 ); // two ints

  return 0;
}
