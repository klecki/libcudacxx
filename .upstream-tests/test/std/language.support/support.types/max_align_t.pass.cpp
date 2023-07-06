//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/type_traits>

// max_align_t is a trivial standard-layout type whose alignment requirement
//   is at least as great as that of every scalar type

#ifndef __CUDACC_RTC__
#include <stdio.h>
#endif // __CUDACC_RTC__
#include "test_macros.h"

#pragma nv_diag_suppress cuda_demote_unsupported_floating_point

int main(int, char**)
{

#if TEST_STD_VER > 17
//  P0767
    static_assert(cuda_for_dali::std::is_trivial<cuda_for_dali::std::max_align_t>::value,
                  "cuda_for_dali::std::is_trivial<cuda_for_dali::std::max_align_t>::value");
    static_assert(cuda_for_dali::std::is_standard_layout<cuda_for_dali::std::max_align_t>::value,
                  "cuda_for_dali::std::is_standard_layout<cuda_for_dali::std::max_align_t>::value");
#else
    static_assert(cuda_for_dali::std::is_pod<cuda_for_dali::std::max_align_t>::value,
                  "cuda_for_dali::std::is_pod<cuda_for_dali::std::max_align_t>::value");
#endif
    static_assert((cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >=
                  cuda_for_dali::std::alignment_of<long long>::value),
                  "cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >= "
                  "cuda_for_dali::std::alignment_of<long long>::value");
    static_assert(cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >=
                  cuda_for_dali::std::alignment_of<long double>::value,
                  "cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >= "
                  "cuda_for_dali::std::alignment_of<long double>::value");
    static_assert(cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >=
                  cuda_for_dali::std::alignment_of<void*>::value,
                  "cuda_for_dali::std::alignment_of<cuda_for_dali::std::max_align_t>::value >= "
                  "cuda_for_dali::std::alignment_of<void*>::value");

  return 0;
}
