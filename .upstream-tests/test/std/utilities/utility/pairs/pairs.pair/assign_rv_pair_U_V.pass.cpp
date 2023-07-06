//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc

// <utility>

// template <class T1, class T2> struct pair

// template<class U, class V> pair& operator=(pair<U, V>&& p);

#include <cuda_for_dali/std/utility>
// cuda/std/memory not supported
// #include <cuda_for_dali/std/memory>
#include <cuda_for_dali/std/cassert>

#include "archetypes.h"
#include "test_macros.h"

struct Base
{
    __host__ __device__ virtual ~Base() {}
};

struct Derived
    : public Base
{
};

int main(int, char**)
{
    // cuda/std/memory not supported
    /*
    {
        typedef cuda_for_dali::std::pair<cuda_for_dali::std::unique_ptr<Derived>, short> P1;
        typedef cuda_for_dali::std::pair<cuda_for_dali::std::unique_ptr<Base>, long> P2;
        P1 p1(cuda_for_dali::std::unique_ptr<Derived>(), static_cast<short>(4));
        P2 p2;
        p2 = cuda_for_dali::std::move(p1);
        assert(p2.first == nullptr);
        assert(p2.second == 4);
    }
    */
    {
       using C = TestTypes::TestType;
       using P = cuda_for_dali::std::pair<int, C>;
       using T = cuda_for_dali::std::pair<long, C>;
       T t(42, -42);
       P p(101, 101);
       C::reset_constructors();
       p = cuda_for_dali::std::move(t);
       assert(C::constructed() == 0);
       assert(C::assigned() == 1);
       assert(C::copy_assigned() == 0);
       assert(C::move_assigned() == 1);
       assert(p.first == 42);
       assert(p.second.value == -42);
    }

  return 0;
}
