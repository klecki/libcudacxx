//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

// overflow should SFINAE instead of error out, LWG 2094

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

__host__ __device__
bool f(cuda_for_dali::std::chrono::milliseconds)
{
    return false;
}

__host__ __device__
bool f(cuda_for_dali::std::chrono::seconds)
{
    return true;
}

int main(int, char**)
{
    {
    cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::exa> r(1);
    assert(f(r));
    }

  return 0;
}
