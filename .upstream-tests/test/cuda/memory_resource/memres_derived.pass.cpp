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
#include <vector>
#include <memory>
#include <tuple>

struct event {
  enum action { ALLOCATE, DEALLOCATE };
  action act;
  std::uintptr_t pointer;
  cuda_for_dali::std::size_t bytes;
  cuda_for_dali::std::size_t alignment;
};

bool operator==(event const& lhs, event const& rhs){
  return std::tie(lhs.act, lhs.pointer, lhs.bytes, lhs.alignment) ==
         std::tie(rhs.act, rhs.pointer, rhs.bytes, rhs.alignment);
}

template <typename MemoryKind>
class derived_resource : public cuda_for_dali::memory_resource<MemoryKind> {
public:
  std::vector<event> &events() { return events_; }
private:
  void *do_allocate(cuda_for_dali::std::size_t bytes,
                    cuda_for_dali::std::size_t alignment) override {
    auto p = 0xDEADBEEF;
    events().push_back(event{event::ALLOCATE, p, bytes, alignment});
    return reinterpret_cast<void*>(p);
  }

  void do_deallocate(void *p, cuda_for_dali::std::size_t bytes,
                     cuda_for_dali::std::size_t alignment) override {
    events().push_back(event{event::DEALLOCATE,
                             reinterpret_cast<std::uintptr_t>(p), bytes,
                             alignment});
  }

  std::vector<event> events_;
};

template <typename MemoryKind>
void test_derived_resource(){
    using derived = derived_resource<MemoryKind>;
    using base = cuda_for_dali::memory_resource<MemoryKind>;

    derived d;
    base * b = &d;

    assert(b->is_equal(*b));
    assert(b->is_equal(d));

    auto p0 = b->allocate(100);
    assert(d.events().size() == 1);
    assert((d.events().back() == event{event::ALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p0),
                                       100, derived::default_alignment}));

    auto p1 = b->allocate(42, 32);
    assert(d.events().size() == 2);
    assert(
        (d.events().back() ==
         event{event::ALLOCATE, reinterpret_cast<std::uintptr_t>(p1), 42, 32}));

    b->deallocate(p0, 100);
    assert(d.events().size() == 3);
    assert((d.events().back() == event{event::DEALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p0),
                                       100, derived::default_alignment}));

    b->deallocate(p1, 42, 32);
    assert(d.events().size() == 4);
    assert((d.events().back() == event{event::DEALLOCATE,
                                       reinterpret_cast<std::uintptr_t>(p1), 42,
                                       32}));
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
  test_derived_resource<cuda_for_dali::memory_kind::host>();
  test_derived_resource<cuda_for_dali::memory_kind::device>();
  test_derived_resource<cuda_for_dali::memory_kind::managed>();
  test_derived_resource<cuda_for_dali::memory_kind::pinned>();
#endif

  return 0;
}
