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

// <cuda/std/barrier>

#include <cuda_for_dali/std/barrier>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda_for_dali::std::barrier<>::max() > 0, "");
  static_assert(cuda_for_dali::std::barrier<void (*)()>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_system>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_system, void (*)()>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_device>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_device, void (*)()>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_block>::max() > 0, "");
  static_assert(cuda_for_dali::barrier<cuda_for_dali::thread_scope_block, void (*)()>::max() > 0, "");
  return 0;
}
