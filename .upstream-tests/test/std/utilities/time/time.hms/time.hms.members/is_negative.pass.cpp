//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

// template <class Duration>
// class hh_mm_ss
//
// constexpr bool is_negative() const noexcept;

#include <cuda_for_dali/std/chrono>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <typename Duration>
__host__ __device__
constexpr bool check_neg(Duration d)
{
	ASSERT_SAME_TYPE(bool, decltype(cuda_for_dali::std::declval<cuda_for_dali::std::chrono::hh_mm_ss<Duration>>().is_negative()));
	ASSERT_NOEXCEPT(                cuda_for_dali::std::declval<cuda_for_dali::std::chrono::hh_mm_ss<Duration>>().is_negative());
	return cuda_for_dali::std::chrono::hh_mm_ss<Duration>(d).is_negative();
}

int main(int, char**)
{
	using microfortnights = cuda_for_dali::std::chrono::duration<int, cuda_for_dali::std::ratio<756, 625>>;

	static_assert(!check_neg(cuda_for_dali::std::chrono::minutes( 1)), "");
	static_assert( check_neg(cuda_for_dali::std::chrono::minutes(-1)), "");

	assert(!check_neg(cuda_for_dali::std::chrono::seconds( 5000)));
	assert( check_neg(cuda_for_dali::std::chrono::seconds(-5000)));
	assert(!check_neg(cuda_for_dali::std::chrono::minutes( 5000)));
	assert( check_neg(cuda_for_dali::std::chrono::minutes(-5000)));
	assert(!check_neg(cuda_for_dali::std::chrono::hours( 11)));
	assert( check_neg(cuda_for_dali::std::chrono::hours(-11)));

	assert(!check_neg(cuda_for_dali::std::chrono::milliseconds( 123456789LL)));
	assert( check_neg(cuda_for_dali::std::chrono::milliseconds(-123456789LL)));
	assert(!check_neg(cuda_for_dali::std::chrono::microseconds( 123456789LL)));
	assert( check_neg(cuda_for_dali::std::chrono::microseconds(-123456789LL)));
	assert(!check_neg(cuda_for_dali::std::chrono::nanoseconds( 123456789LL)));
	assert( check_neg(cuda_for_dali::std::chrono::nanoseconds(-123456789LL)));

	assert(!check_neg(microfortnights( 10000)));
	assert( check_neg(microfortnights(-10000)));

	return 0;
}
