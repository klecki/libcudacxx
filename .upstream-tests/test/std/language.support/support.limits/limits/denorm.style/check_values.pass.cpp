//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// float_round_style

#include <cuda_for_dali/std/limits>

#include "test_macros.h"

typedef char one;
struct two {one _[2];};

__host__ __device__ one test(cuda_for_dali::std::float_round_style);
__host__ __device__ two test(int);

int main(int, char**)
{
    static_assert(cuda_for_dali::std::round_indeterminate == -1,
                 "cuda_for_dali::std::round_indeterminate == -1");
    static_assert(cuda_for_dali::std::round_toward_zero == 0,
                 "cuda_for_dali::std::round_toward_zero == 0");
    static_assert(cuda_for_dali::std::round_to_nearest == 1,
                 "cuda_for_dali::std::round_to_nearest == 1");
    static_assert(cuda_for_dali::std::round_toward_infinity == 2,
                 "cuda_for_dali::std::round_toward_infinity == 2");
    static_assert(cuda_for_dali::std::round_toward_neg_infinity == 3,
                 "cuda_for_dali::std::round_toward_neg_infinity == 3");
    static_assert(sizeof(test(cuda_for_dali::std::round_to_nearest)) == 1,
                 "sizeof(test(cuda_for_dali::std::round_to_nearest)) == 1");
    static_assert(sizeof(test(1)) == 2,
                 "sizeof(test(1)) == 2");

  return 0;
}
