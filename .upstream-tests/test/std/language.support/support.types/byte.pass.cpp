//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/type_traits>
#include "test_macros.h"

// XFAIL: c++98, c++03, c++11

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// cuda_for_dali::std::byte is not an integer type, nor a character type.
// It is a distinct type for accessing the bits that ultimately make up object storage.

#if TEST_STD_VER > 11
static_assert( cuda_for_dali::std::is_trivial<cuda_for_dali::std::byte>::value, "" );   // P0767
#else
static_assert( cuda_for_dali::std::is_pod<cuda_for_dali::std::byte>::value, "" );
#endif
static_assert(!cuda_for_dali::std::is_arithmetic<cuda_for_dali::std::byte>::value, "" );
static_assert(!cuda_for_dali::std::is_integral<cuda_for_dali::std::byte>::value, "" );

static_assert(!cuda_for_dali::std::is_same<cuda_for_dali::std::byte,          char>::value, "" );
static_assert(!cuda_for_dali::std::is_same<cuda_for_dali::std::byte,   signed char>::value, "" );
static_assert(!cuda_for_dali::std::is_same<cuda_for_dali::std::byte, unsigned char>::value, "" );

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(cuda_for_dali::std::byte) == 1, "" );

int main(int, char**) {
  return 0;
}
