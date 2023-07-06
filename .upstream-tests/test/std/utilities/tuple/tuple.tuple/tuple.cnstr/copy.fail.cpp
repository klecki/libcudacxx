//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple(const tuple& u) = default;

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: nvrtc

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "MoveOnly.h"

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::tuple<MoveOnly> T;
        T t0(MoveOnly(2));
        T t = t0;
    }

  return 0;
}
