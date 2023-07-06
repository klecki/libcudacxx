//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test fails because diagnose_if doesn't emit all of the diagnostics
// when -fdelayed-template-parsing is enabled, like it is on Windows.
// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: verify-support, diagnose-if-support
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/std/atomic>

// Test that invalid memory order arguments are diagnosed where possible.

#include <cuda_for_dali/std/atomic>

int main(int, char**) {
    cuda_for_dali::std::atomic<int> x(42);
    volatile cuda_for_dali::std::atomic<int>& vx = x;
    int val1 = 1; ((void)val1);
    int val2 = 2; ((void)val2);
    // load operations
    {
        x.load(cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.load(cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.load(cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.load(cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.load(cuda_for_dali::std::memory_order_relaxed);
        x.load(cuda_for_dali::std::memory_order_consume);
        x.load(cuda_for_dali::std::memory_order_acquire);
        x.load(cuda_for_dali::std::memory_order_seq_cst);
    }
    {
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_load_explicit(&vx, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_load_explicit(&vx, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_relaxed);
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_consume);
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_acquire);
        cuda_for_dali::std::atomic_load_explicit(&x, cuda_for_dali::std::memory_order_seq_cst);
    }
    // store operations
    {
        x.store(42, cuda_for_dali::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.store(42, cuda_for_dali::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.store(42, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, cuda_for_dali::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, cuda_for_dali::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.store(42, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.store(42, cuda_for_dali::std::memory_order_relaxed);
        x.store(42, cuda_for_dali::std::memory_order_release);
        x.store(42, cuda_for_dali::std::memory_order_seq_cst);
    }
    {
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_store_explicit(&vx, 42, cuda_for_dali::std::memory_order_consume); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_store_explicit(&vx, 42, cuda_for_dali::std::memory_order_acquire); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_store_explicit(&vx, 42, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_relaxed);
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_release);
        cuda_for_dali::std::atomic_store_explicit(&x, 42, cuda_for_dali::std::memory_order_seq_cst);
    }
    // compare exchange weak
    {
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_relaxed);
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_consume);
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acquire);
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_seq_cst);
        // Test that the cmpxchg overload with only one memory order argument
        // does not generate any diagnostics.
        x.compare_exchange_weak(val1, val2, cuda_for_dali::std::memory_order_release);
    }
    {
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&vx, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&vx, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_relaxed);
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_consume);
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acquire);
        cuda_for_dali::std::atomic_compare_exchange_weak_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_seq_cst);
    }
    // compare exchange strong
    {
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        vx.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_relaxed);
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_consume);
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acquire);
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_seq_cst);
        // Test that the cmpxchg overload with only one memory order argument
        // does not generate any diagnostics.
        x.compare_exchange_strong(val1, val2, cuda_for_dali::std::memory_order_release);
    }
    {
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&vx, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_release); // expected-warning {{memory order argument to atomic operation is invalid}}
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&vx, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acq_rel); // expected-warning {{memory order argument to atomic operation is invalid}}
        // valid memory orders
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_relaxed);
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_consume);
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_acquire);
        cuda_for_dali::std::atomic_compare_exchange_strong_explicit(&x, &val1, val2, cuda_for_dali::std::memory_order_seq_cst, cuda_for_dali::std::memory_order_seq_cst);
    }

  return 0;
}
