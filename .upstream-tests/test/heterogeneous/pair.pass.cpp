//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// uncomment for a really verbose output detailing what test steps are being launched
// #define DEBUG_TESTERS

#include "helpers.h"

#include <cuda_for_dali/std/utility>
#include <cuda_for_dali/std/cassert>

struct pod {
    char val[10];
};

using pair_t = cuda_for_dali::std::pair<int, pod>;

template<int N>
struct write
{
    using async = cuda_for_dali::std::false_type;

    template <typename Pair>
    __host__ __device__
    static void perform(Pair &p)
    {
        cuda_for_dali::std::get<0>(p) = N;
        cuda_for_dali::std::get<1>(p).val[0] = N;
    }
};

template<int N>
struct read
{
    using async = cuda_for_dali::std::false_type;

    template <typename Pair>
    __host__ __device__
    static void perform(Pair &p)
    {
        assert(cuda_for_dali::std::get<0>(p) == N);
        assert(cuda_for_dali::std::get<1>(p).val[0] == N);
    }
};

using w_r_w_r = performer_list<
  write<10>,
  read<10>,
  write<30>,
  read<30>
>;

void kernel_invoker()
{
    pair_t p(0, {0});
    validate_not_movable<
        pair_t,
        w_r_w_r
    >(p);
}

int main(int arg, char ** argv)
{
#ifndef __CUDA_ARCH__
    kernel_invoker();
#endif

    return 0;
}
