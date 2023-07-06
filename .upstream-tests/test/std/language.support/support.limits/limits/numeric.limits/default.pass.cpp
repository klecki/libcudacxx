//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// The default numeric_limits<T> template shall have all members, but with
// 0 or false values.

#include <cuda_for_dali/std/limits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

struct A
{
    __host__ __device__
    A(int i = 0) : data_(i) {}
    int data_;
};

__host__ __device__
bool operator == (const A& x, const A& y) {return x.data_ == y.data_;}

int main(int, char**)
{
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_specialized == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_specialized == false");
    assert(cuda_for_dali::std::numeric_limits<A>::min() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::max() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::lowest() == A());
    static_assert(cuda_for_dali::std::numeric_limits<A>::digits == 0,
                 "cuda_for_dali::std::numeric_limits<A>::digits == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::digits10 == 0,
                 "cuda_for_dali::std::numeric_limits<A>::digits10 == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::max_digits10 == 0,
                 "cuda_for_dali::std::numeric_limits<A>::max_digits10 == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_signed == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_signed == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_integer == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_integer == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_exact == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_exact == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::radix == 0,
                 "cuda_for_dali::std::numeric_limits<A>::radix == 0");
    assert(cuda_for_dali::std::numeric_limits<A>::epsilon() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::round_error() == A());
    static_assert(cuda_for_dali::std::numeric_limits<A>::min_exponent == 0,
                 "cuda_for_dali::std::numeric_limits<A>::min_exponent == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::min_exponent10 == 0,
                 "cuda_for_dali::std::numeric_limits<A>::min_exponent10 == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::max_exponent == 0,
                 "cuda_for_dali::std::numeric_limits<A>::max_exponent == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::max_exponent10 == 0,
                 "cuda_for_dali::std::numeric_limits<A>::max_exponent10 == 0");
    static_assert(cuda_for_dali::std::numeric_limits<A>::has_infinity == false,
                 "cuda_for_dali::std::numeric_limits<A>::has_infinity == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::has_quiet_NaN == false,
                 "cuda_for_dali::std::numeric_limits<A>::has_quiet_NaN == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::has_signaling_NaN == false,
                 "cuda_for_dali::std::numeric_limits<A>::has_signaling_NaN == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::has_denorm == cuda_for_dali::std::denorm_absent,
                 "cuda_for_dali::std::numeric_limits<A>::has_denorm == cuda_for_dali::std::denorm_absent");
    static_assert(cuda_for_dali::std::numeric_limits<A>::has_denorm_loss == false,
                 "cuda_for_dali::std::numeric_limits<A>::has_denorm_loss == false");
    assert(cuda_for_dali::std::numeric_limits<A>::infinity() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::quiet_NaN() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::signaling_NaN() == A());
    assert(cuda_for_dali::std::numeric_limits<A>::denorm_min() == A());
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_iec559 == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_iec559 == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_bounded == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_bounded == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::is_modulo == false,
                 "cuda_for_dali::std::numeric_limits<A>::is_modulo == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::traps == false,
                 "cuda_for_dali::std::numeric_limits<A>::traps == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::tinyness_before == false,
                 "cuda_for_dali::std::numeric_limits<A>::tinyness_before == false");
    static_assert(cuda_for_dali::std::numeric_limits<A>::round_style == cuda_for_dali::std::round_toward_zero,
                 "cuda_for_dali::std::numeric_limits<A>::round_style == cuda_for_dali::std::round_toward_zero");

  return 0;
}
