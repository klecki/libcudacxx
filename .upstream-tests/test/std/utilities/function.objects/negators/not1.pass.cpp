//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// not1

#define _LIBCUDAFORDALICXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda_for_dali/std/functional>
#include <cuda_for_dali/std/cassert>

int main(int, char**)
{
    typedef cuda_for_dali::std::logical_not<int> F;
    assert(cuda_for_dali::std::not1(F())(36));
    assert(!cuda_for_dali::std::not1(F())(0));

  return 0;
}
