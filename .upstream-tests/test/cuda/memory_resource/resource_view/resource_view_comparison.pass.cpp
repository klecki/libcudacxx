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
#include <memory>
#include <tuple>
#include <vector>


template <typename tag>
class resource : public cuda_for_dali::stream_ordered_memory_resource<cuda_for_dali::memory_kind::managed> {
public:
  int value = 0;
private:
  void *do_allocate(size_t, size_t) override {
    return nullptr;
  }

  void *do_allocate_async(size_t, size_t, cuda_for_dali::stream_view) override {
    return nullptr;
  }

  void do_deallocate(void *, size_t, size_t) {
  }

  void do_deallocate_async(void *, size_t, size_t, cuda_for_dali::stream_view) override {
  }

#ifdef _LIBCUDAFORDALICXX_EXT_RTTI_ENABLED
  bool do_is_equal(const cuda_for_dali::memory_resource<cuda_for_dali::memory_kind::managed> &other) const noexcept override {
    fprintf(stderr, "Comparison start: %p %p\n", this, &other);
    if (auto *other_ptr = dynamic_cast<const resource *>(&other)) {
      fprintf(stderr, "values: %d %d\n", value, other_ptr->value);
      return value == other_ptr->value;
    } else {
      return false;
    }
  }
#endif
};


struct tag1;
struct tag2;


int main(int argc, char **argv) {
#if !defined(__CUDA_ARCH__) && defined(_LIBCUDAFORDALICXX_EXT_RTTI_ENABLED)
  resource<tag1> r1, r2, r3;
  resource<tag2> r4;
  cuda_for_dali::basic_resource_view<resource<tag1>*, cuda_for_dali::is_kind<cuda_for_dali::memory_kind::managed>> v1_null;
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host> v2_null;
  assert(v1_null == v2_null);
  r1.value = 42;
  r2.value = 42;
  r3.value = 99;
  r4.value = 42;
  using t1 = decltype(view_resource(&r1));
  using t2 = decltype(view_resource(&r2));
  using t4 = decltype(view_resource(&r4));
  assert(view_resource(&r1) == view_resource(&r2));
  assert(view_resource(&r1) != view_resource(&r3));
  assert(view_resource(&r4) == view_resource(&r4));
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host, cuda_for_dali::memory_access::device>   v1 = &r1;
  cuda_for_dali::stream_ordered_resource_view<cuda_for_dali::memory_access::device> v2 = &r2;
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::host> v3 = &r3;
  cuda_for_dali::resource_view<cuda_for_dali::memory_access::device, cuda_for_dali::memory_access::host> v4 = &r4;
  // compare views
  assert(v1 == v2);
  assert(v1 != v3);
  assert(v1 != v4);
  // assert(v2 != v3); - cannot compare - incompatible views
  assert(v2 != v4);
  assert(v3 != v4);
  assert(v4 == v4);

  // compare views vs resources
  assert(v1 == &r1);
  assert(v1 == &r2);
  assert(v1 != &r4);
  assert(v2 == &r2);
  assert(v2 != &r4);
  assert(v4 == &r4);

  assert(&r1 == v1);
  assert(&r2 == v1);
  assert(&r4 != v1);
  assert(&r2 == v2);
  assert(&r4 != v2);
  assert(&r4 == v4);
#endif
  return 0;
}
