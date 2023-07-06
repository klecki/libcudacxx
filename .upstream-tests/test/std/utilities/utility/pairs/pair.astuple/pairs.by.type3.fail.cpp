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
    // cuda/std/memory not supported, however, with <complex> available this test needs to fail.
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    cuda_for_dali::std::pair<upint, int> t(upint(new int(4)), 23);
    upint p = cuda_for_dali::std::get<upint>(t);

  return 0;
}
