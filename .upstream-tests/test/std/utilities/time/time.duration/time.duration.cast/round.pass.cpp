//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <cuda/std/chrono>

// round

// template <class ToDuration, class Rep, class Period>
//   constexpr
//   ToDuration
//   ceil(const duration<Rep, Period>& d);

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

template <class ToDuration, class FromDuration>
__host__ __device__
void
test(const FromDuration& f, const ToDuration& d)
{
    {
    typedef decltype(cuda_for_dali::std::chrono::round<ToDuration>(f)) R;
    static_assert((cuda_for_dali::std::is_same<R, ToDuration>::value), "");
    assert(cuda_for_dali::std::chrono::round<ToDuration>(f) == d);
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(cuda_for_dali::std::chrono::milliseconds( 7290000), cuda_for_dali::std::chrono::hours( 2));
    test(cuda_for_dali::std::chrono::milliseconds(-7290000), cuda_for_dali::std::chrono::hours(-2));
    test(cuda_for_dali::std::chrono::milliseconds( 7290000), cuda_for_dali::std::chrono::minutes( 122));
    test(cuda_for_dali::std::chrono::milliseconds(-7290000), cuda_for_dali::std::chrono::minutes(-122));

    {
//  9000000ms is 2 hours and 30 minutes
    constexpr cuda_for_dali::std::chrono::hours h1 = cuda_for_dali::std::chrono::round<cuda_for_dali::std::chrono::hours>(cuda_for_dali::std::chrono::milliseconds(9000000));
    static_assert(h1.count() == 2, "");
    constexpr cuda_for_dali::std::chrono::hours h2 = cuda_for_dali::std::chrono::round<cuda_for_dali::std::chrono::hours>(cuda_for_dali::std::chrono::milliseconds(-9000000));
    static_assert(h2.count() == -2, "");
    }

  return 0;
}
