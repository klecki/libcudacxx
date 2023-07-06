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
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/stream_view>

template <typename Kind> constexpr bool test_memory_kind() {
  using mr = cuda_for_dali::stream_ordered_memory_resource<Kind>;
  return std::is_same<typename mr::memory_kind, Kind>::value;
}

template <typename Kind, std::size_t Alignment>
constexpr bool test_alignment() {
  using mr = cuda_for_dali::stream_ordered_memory_resource<Kind>;
  return mr::default_alignment == Alignment;
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  namespace memory_kind = cuda_for_dali::memory_kind;
  static_assert(test_memory_kind<memory_kind::host>(), "");
  static_assert(test_memory_kind<memory_kind::device>(), "");
  static_assert(test_memory_kind<memory_kind::managed>(), "");
  static_assert(test_memory_kind<memory_kind::pinned>(), "");

  using mr = cuda_for_dali::stream_ordered_memory_resource<memory_kind::host>;

  static_assert(test_alignment<memory_kind::host, alignof(cuda_for_dali::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::device, alignof(cuda_for_dali::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::managed, alignof(cuda_for_dali::std::max_align_t)>(), "");
  static_assert(test_alignment<memory_kind::pinned, alignof(cuda_for_dali::std::max_align_t)>(), "");
#endif

  return 0;
}
