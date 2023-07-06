//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template<Returnable R, class T> unspecified mem_fn(R T::* pm);

#include <cuda_for_dali/std/functional>
#include <cuda_for_dali/std/cassert>

struct A
{
    double data_;
};

template <class F>
__host__ __device__
void
test(F f)
{
    {
    A a;
    f(a) = 5;
    assert(a.data_ == 5);
    A* ap = &a;
    f(ap) = 6;
    assert(a.data_ == 6);
    const A* cap = ap;
    assert(f(cap) == f(ap));
    const F& cf = f;
    assert(cf(ap) == f(ap));
    }
}

int main(int, char**)
{
    test(cuda_for_dali::std::mem_fn(&A::data_));

  return 0;
}
