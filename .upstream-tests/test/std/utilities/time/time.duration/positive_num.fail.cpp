//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// Period::num shall be positive, diagnostic required.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <cuda_for_dali/std/chrono>

int main(int, char**)
{
    typedef cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<5, -1> > D;
    D d;

  return 0;
}
