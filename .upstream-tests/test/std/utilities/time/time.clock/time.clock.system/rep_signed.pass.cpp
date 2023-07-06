//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// system_clock

// rep should be signed

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

int main(int, char**)
{
    assert(cuda_for_dali::std::chrono::system_clock::duration::min() <
           cuda_for_dali::std::chrono::system_clock::duration::zero());

  return 0;
}
