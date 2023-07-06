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

// <cuda/std/atomic>

// struct atomic_flag

// bool test_and_set(memory_order = memory_order_seq_cst);
// bool test_and_set(memory_order = memory_order_seq_cst) volatile;

#include <cuda_for_dali/std/atomic>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "cuda_space_selector.h"

template<template<typename, typename> class Selector>
__host__ __device__
void test()
{
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        assert(f.test_and_set() == 1);
    }
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_relaxed) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_relaxed) == 1);
    }
#ifndef __INTEL_COMPILER
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_consume) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_consume) == 1);
    }
#endif
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acquire) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acquire) == 1);
    }
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_release) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_release) == 1);
    }
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acq_rel) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acq_rel) == 1);
    }
    {
        Selector<cuda_for_dali::std::atomic_flag, default_initializer> sel;
        cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_seq_cst) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_seq_cst) == 1);
    }
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set() == 0);
        assert(f.test_and_set() == 1);
    }
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_relaxed) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_relaxed) == 1);
    }
#ifndef __INTEL_COMPILER
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_consume) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_consume) == 1);
    }
#endif
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acquire) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acquire) == 1);
    }
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_release) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_release) == 1);
    }
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acq_rel) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_acq_rel) == 1);
    }
    {
        Selector<volatile cuda_for_dali::std::atomic_flag, default_initializer> sel;
        volatile cuda_for_dali::std::atomic_flag & f = *sel.construct();
        f.clear();
        assert(f.test_and_set(cuda_for_dali::std::memory_order_seq_cst) == 0);
        assert(f.test_and_set(cuda_for_dali::std::memory_order_seq_cst) == 1);
    }
}

int main(int, char**)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    test<local_memory_selector>();
#endif
#ifdef __CUDA_ARCH__
    test<shared_memory_selector>();
    test<global_memory_selector>();
#endif

  return 0;
}
