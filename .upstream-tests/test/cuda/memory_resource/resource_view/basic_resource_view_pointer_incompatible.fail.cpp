//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cuda_for_dali/memory_resource>
#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/stream_view>
#include <memory>
#include <tuple>
#include <vector>
#include "resource_hierarchy.h"

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  {
    view_D2 da2;
    view_MA ma = da2;  // cant't assign syncrhounous to asyncrhounous resource
  }
#endif
  return 0;
}
