//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class ...UTypes>
//   tuple(allocator_arg_t, const Alloc& a, tuple<UTypes...>&&);

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: nvrtc

#include <cuda_for_dali/std/tuple>

struct ExplicitCopy {
  __host__ __device__ explicit ExplicitCopy(int) {}
  __host__ __device__ explicit ExplicitCopy(ExplicitCopy const&) {}
};

__host__ __device__  cuda_for_dali::std::tuple<ExplicitCopy> explicit_move_test() {
    cuda_for_dali::std::tuple<int> t1(42);
    return {cuda_for_dali::std::allocator_arg, cuda_for_dali::std::allocator<void>{}, cuda_for_dali::std::move(t1)};
    // expected-error@-1 {{chosen constructor is explicit in copy-initialization}}
}

int main(int, char**)
{


  return 0;
}
