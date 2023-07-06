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

int main(int argc, char **argv) {
#ifndef __CUDA_ARCH__
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host,
                      cuda_for_dali::oversubscribable,
                      cuda_for_dali::memory_location::host> props_only;

  // test copy constructor
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host,
                      cuda_for_dali::oversubscribable,
                      cuda_for_dali::memory_location::host> same_props = props_only;

  // test converting constructor
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host,
                      cuda_for_dali::oversubscribable,
                      cuda_for_dali::memory_location::host> subset = props_only;

  // test asisgnment
  same_props = props_only;
  subset = props_only;
#endif
  return 0;
}
