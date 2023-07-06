//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... assertion fails line 35

// <cuda/std/atomic>

// template <class T>
//     T
//     atomic_load(const volatile atomic<T>* obj);
//
// template <class T>
//     T
//     atomic_load(const atomic<T>* obj);

#include <cuda_for_dali/std/atomic>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "cuda_space_selector.h"

template <class T, template<typename, typename> typename Selector, cuda_for_dali::thread_scope>
struct TestFn {
  __host__ __device__
  void operator()() const {
    typedef cuda_for_dali::std::atomic<T> A;
    Selector<A, constructor_initializer> sel;
    A & t = *sel.construct();
    cuda_for_dali::std::atomic_init(&t, T(1));
    assert(cuda_for_dali::std::atomic_load(&t) == T(1));
    Selector<volatile A, constructor_initializer> vsel;
    volatile A & vt = *vsel.construct();
    cuda_for_dali::std::atomic_init(&vt, T(2));
    assert(cuda_for_dali::std::atomic_load(&vt) == T(2));
  }
};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    TestEachAtomicType<TestFn, local_memory_selector>()();
#endif
#ifdef __CUDA_ARCH__
    TestEachAtomicType<TestFn, shared_memory_selector>()();
    TestEachAtomicType<TestFn, global_memory_selector>()();
#endif

  return 0;
}
