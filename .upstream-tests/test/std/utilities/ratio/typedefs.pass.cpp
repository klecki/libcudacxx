//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio typedef's

#include <cuda_for_dali/std/ratio>

int main(int, char**)
{
    static_assert(cuda_for_dali::std::atto::num == 1 && cuda_for_dali::std::atto::den == 1000000000000000000ULL, "");
    static_assert(cuda_for_dali::std::femto::num == 1 && cuda_for_dali::std::femto::den == 1000000000000000ULL, "");
    static_assert(cuda_for_dali::std::pico::num == 1 && cuda_for_dali::std::pico::den == 1000000000000ULL, "");
    static_assert(cuda_for_dali::std::nano::num == 1 && cuda_for_dali::std::nano::den == 1000000000ULL, "");
    static_assert(cuda_for_dali::std::micro::num == 1 && cuda_for_dali::std::micro::den == 1000000ULL, "");
    static_assert(cuda_for_dali::std::milli::num == 1 && cuda_for_dali::std::milli::den == 1000ULL, "");
    static_assert(cuda_for_dali::std::centi::num == 1 && cuda_for_dali::std::centi::den == 100ULL, "");
    static_assert(cuda_for_dali::std::deci::num == 1 && cuda_for_dali::std::deci::den == 10ULL, "");
    static_assert(cuda_for_dali::std::deca::num == 10ULL && cuda_for_dali::std::deca::den == 1, "");
    static_assert(cuda_for_dali::std::hecto::num == 100ULL && cuda_for_dali::std::hecto::den == 1, "");
    static_assert(cuda_for_dali::std::kilo::num == 1000ULL && cuda_for_dali::std::kilo::den == 1, "");
    static_assert(cuda_for_dali::std::mega::num == 1000000ULL && cuda_for_dali::std::mega::den == 1, "");
    static_assert(cuda_for_dali::std::giga::num == 1000000000ULL && cuda_for_dali::std::giga::den == 1, "");
    static_assert(cuda_for_dali::std::tera::num == 1000000000000ULL && cuda_for_dali::std::tera::den == 1, "");
    static_assert(cuda_for_dali::std::peta::num == 1000000000000000ULL && cuda_for_dali::std::peta::den == 1, "");
    static_assert(cuda_for_dali::std::exa::num == 1000000000000000000ULL && cuda_for_dali::std::exa::den == 1, "");

  return 0;
}
