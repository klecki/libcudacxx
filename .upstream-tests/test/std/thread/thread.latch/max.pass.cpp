//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/std/latch>

#include <cuda_for_dali/std/latch>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda_for_dali::std::latch::max() > 0, "");
  static_assert(cuda_for_dali::latch<cuda_for_dali::thread_scope_system>::max() > 0, "");
  static_assert(cuda_for_dali::latch<cuda_for_dali::thread_scope_device>::max() > 0, "");
  static_assert(cuda_for_dali::latch<cuda_for_dali::thread_scope_block>::max() > 0, "");
  return 0;
}
