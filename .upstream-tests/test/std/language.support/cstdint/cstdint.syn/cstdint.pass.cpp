//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <cuda/std/cstdint>

#include <cuda_for_dali/std/cstdint>
#include <cuda_for_dali/std/cstddef>
// #include <cuda_for_dali/std/cwchar>
// #include <cuda_for_dali/std/csignal>
// #include <cuda_for_dali/std/cwctype>
#include <cuda_for_dali/std/climits>
#include <cuda_for_dali/std/type_traits>
// #include <cuda_for_dali/std/limits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // typedef cuda_for_dali::std::int8_t
    static_assert(sizeof(cuda_for_dali::std::int8_t)*CHAR_BIT == 8,
                 "sizeof(cuda_for_dali::std::int8_t)*CHAR_BIT == 8");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int8_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int8_t>::value");
    // typedef cuda_for_dali::std::int16_t
    static_assert(sizeof(cuda_for_dali::std::int16_t)*CHAR_BIT == 16,
                 "sizeof(cuda_for_dali::std::int16_t)*CHAR_BIT == 16");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int16_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int16_t>::value");
    // typedef cuda_for_dali::std::int32_t
    static_assert(sizeof(cuda_for_dali::std::int32_t)*CHAR_BIT == 32,
                 "sizeof(cuda_for_dali::std::int32_t)*CHAR_BIT == 32");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int32_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int32_t>::value");
    // typedef cuda_for_dali::std::int64_t
    static_assert(sizeof(cuda_for_dali::std::int64_t)*CHAR_BIT == 64,
                 "sizeof(cuda_for_dali::std::int64_t)*CHAR_BIT == 64");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int64_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int64_t>::value");

    // typedef cuda_for_dali::std::uint8_t
    static_assert(sizeof(cuda_for_dali::std::uint8_t)*CHAR_BIT == 8,
                 "sizeof(cuda_for_dali::std::uint8_t)*CHAR_BIT == 8");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint8_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint8_t>::value");
    // typedef cuda_for_dali::std::uint16_t
    static_assert(sizeof(cuda_for_dali::std::uint16_t)*CHAR_BIT == 16,
                 "sizeof(cuda_for_dali::std::uint16_t)*CHAR_BIT == 16");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint16_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint16_t>::value");
    // typedef cuda_for_dali::std::uint32_t
    static_assert(sizeof(cuda_for_dali::std::uint32_t)*CHAR_BIT == 32,
                 "sizeof(cuda_for_dali::std::uint32_t)*CHAR_BIT == 32");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint32_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint32_t>::value");
    // typedef cuda_for_dali::std::uint64_t
    static_assert(sizeof(cuda_for_dali::std::uint64_t)*CHAR_BIT == 64,
                 "sizeof(cuda_for_dali::std::uint64_t)*CHAR_BIT == 64");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint64_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint64_t>::value");

    // typedef cuda_for_dali::std::int_least8_t
    static_assert(sizeof(cuda_for_dali::std::int_least8_t)*CHAR_BIT >= 8,
                 "sizeof(cuda_for_dali::std::int_least8_t)*CHAR_BIT >= 8");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least8_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least8_t>::value");
    // typedef cuda_for_dali::std::int_least16_t
    static_assert(sizeof(cuda_for_dali::std::int_least16_t)*CHAR_BIT >= 16,
                 "sizeof(cuda_for_dali::std::int_least16_t)*CHAR_BIT >= 16");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least16_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least16_t>::value");
    // typedef cuda_for_dali::std::int_least32_t
    static_assert(sizeof(cuda_for_dali::std::int_least32_t)*CHAR_BIT >= 32,
                 "sizeof(cuda_for_dali::std::int_least32_t)*CHAR_BIT >= 32");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least32_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least32_t>::value");
    // typedef cuda_for_dali::std::int_least64_t
    static_assert(sizeof(cuda_for_dali::std::int_least64_t)*CHAR_BIT >= 64,
                 "sizeof(cuda_for_dali::std::int_least64_t)*CHAR_BIT >= 64");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least64_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_least64_t>::value");

    // typedef cuda_for_dali::std::uint_least8_t
    static_assert(sizeof(cuda_for_dali::std::uint_least8_t)*CHAR_BIT >= 8,
                 "sizeof(cuda_for_dali::std::uint_least8_t)*CHAR_BIT >= 8");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least8_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least8_t>::value");
    // typedef cuda_for_dali::std::uint_least16_t
    static_assert(sizeof(cuda_for_dali::std::uint_least16_t)*CHAR_BIT >= 16,
                 "sizeof(cuda_for_dali::std::uint_least16_t)*CHAR_BIT >= 16");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least16_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least16_t>::value");
    // typedef cuda_for_dali::std::uint_least32_t
    static_assert(sizeof(cuda_for_dali::std::uint_least32_t)*CHAR_BIT >= 32,
                 "sizeof(cuda_for_dali::std::uint_least32_t)*CHAR_BIT >= 32");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least32_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least32_t>::value");
    // typedef cuda_for_dali::std::uint_least64_t
    static_assert(sizeof(cuda_for_dali::std::uint_least64_t)*CHAR_BIT >= 64,
                 "sizeof(cuda_for_dali::std::uint_least64_t)*CHAR_BIT >= 64");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least64_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_least64_t>::value");

    // typedef cuda_for_dali::std::int_fast8_t
    static_assert(sizeof(cuda_for_dali::std::int_fast8_t)*CHAR_BIT >= 8,
                 "sizeof(cuda_for_dali::std::int_fast8_t)*CHAR_BIT >= 8");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast8_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast8_t>::value");
    // typedef cuda_for_dali::std::int_fast16_t
    static_assert(sizeof(cuda_for_dali::std::int_fast16_t)*CHAR_BIT >= 16,
                 "sizeof(cuda_for_dali::std::int_fast16_t)*CHAR_BIT >= 16");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast16_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast16_t>::value");
    // typedef cuda_for_dali::std::int_fast32_t
    static_assert(sizeof(cuda_for_dali::std::int_fast32_t)*CHAR_BIT >= 32,
                 "sizeof(cuda_for_dali::std::int_fast32_t)*CHAR_BIT >= 32");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast32_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast32_t>::value");
    // typedef cuda_for_dali::std::int_fast64_t
    static_assert(sizeof(cuda_for_dali::std::int_fast64_t)*CHAR_BIT >= 64,
                 "sizeof(cuda_for_dali::std::int_fast64_t)*CHAR_BIT >= 64");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast64_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::int_fast64_t>::value");

    // typedef cuda_for_dali::std::uint_fast8_t
    static_assert(sizeof(cuda_for_dali::std::uint_fast8_t)*CHAR_BIT >= 8,
                 "sizeof(cuda_for_dali::std::uint_fast8_t)*CHAR_BIT >= 8");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast8_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast8_t>::value");
    // typedef cuda_for_dali::std::uint_fast16_t
    static_assert(sizeof(cuda_for_dali::std::uint_fast16_t)*CHAR_BIT >= 16,
                 "sizeof(cuda_for_dali::std::uint_fast16_t)*CHAR_BIT >= 16");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast16_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast16_t>::value");
    // typedef cuda_for_dali::std::uint_fast32_t
    static_assert(sizeof(cuda_for_dali::std::uint_fast32_t)*CHAR_BIT >= 32,
                 "sizeof(cuda_for_dali::std::uint_fast32_t)*CHAR_BIT >= 32");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast32_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast32_t>::value");
    // typedef cuda_for_dali::std::uint_fast64_t
    static_assert(sizeof(cuda_for_dali::std::uint_fast64_t)*CHAR_BIT >= 64,
                 "sizeof(cuda_for_dali::std::uint_fast64_t)*CHAR_BIT >= 64");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast64_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uint_fast64_t>::value");

    // typedef cuda_for_dali::std::intptr_t
    static_assert(sizeof(cuda_for_dali::std::intptr_t) >= sizeof(void*),
                 "sizeof(cuda_for_dali::std::intptr_t) >= sizeof(void*)");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::intptr_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::intptr_t>::value");
    // typedef cuda_for_dali::std::uintptr_t
    static_assert(sizeof(cuda_for_dali::std::uintptr_t) >= sizeof(void*),
                 "sizeof(cuda_for_dali::std::uintptr_t) >= sizeof(void*)");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uintptr_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uintptr_t>::value");

    // typedef cuda_for_dali::std::intmax_t
    static_assert(sizeof(cuda_for_dali::std::intmax_t) >= sizeof(long long),
                 "sizeof(cuda_for_dali::std::intmax_t) >= sizeof(long long)");
    static_assert(cuda_for_dali::std::is_signed<cuda_for_dali::std::intmax_t>::value,
                 "cuda_for_dali::std::is_signed<cuda_for_dali::std::intmax_t>::value");
    // typedef cuda_for_dali::std::uintmax_t
    static_assert(sizeof(cuda_for_dali::std::uintmax_t) >= sizeof(unsigned long long),
                 "sizeof(cuda_for_dali::std::uintmax_t) >= sizeof(unsigned long long)");
    static_assert(cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uintmax_t>::value,
                 "cuda_for_dali::std::is_unsigned<cuda_for_dali::std::uintmax_t>::value");

    // INTN_MIN
    static_assert(INT8_MIN == -128, "INT8_MIN == -128");
    static_assert(INT16_MIN == -32768, "INT16_MIN == -32768");
    static_assert(INT32_MIN == -2147483647 - 1, "INT32_MIN == -2147483648");
    static_assert(INT64_MIN == -9223372036854775807LL - 1, "INT64_MIN == -9223372036854775808LL");

    // INTN_MAX
    static_assert(INT8_MAX == 127, "INT8_MAX == 127");
    static_assert(INT16_MAX == 32767, "INT16_MAX == 32767");
    static_assert(INT32_MAX == 2147483647, "INT32_MAX == 2147483647");
    static_assert(INT64_MAX == 9223372036854775807LL, "INT64_MAX == 9223372036854775807LL");

    // UINTN_MAX
    static_assert(UINT8_MAX == 255, "UINT8_MAX == 255");
    static_assert(UINT16_MAX == 65535, "UINT16_MAX == 65535");
    static_assert(UINT32_MAX == 4294967295U, "UINT32_MAX == 4294967295");
    static_assert(UINT64_MAX == 18446744073709551615ULL, "UINT64_MAX == 18446744073709551615ULL");

    // INT_FASTN_MIN
    static_assert(INT_FAST8_MIN <= -128, "INT_FAST8_MIN <= -128");
    static_assert(INT_FAST16_MIN <= -32768, "INT_FAST16_MIN <= -32768");
    static_assert(INT_FAST32_MIN <= -2147483647 - 1, "INT_FAST32_MIN <= -2147483648");
    static_assert(INT_FAST64_MIN <= -9223372036854775807LL - 1, "INT_FAST64_MIN <= -9223372036854775808LL");

    // INT_FASTN_MAX
    static_assert(INT_FAST8_MAX >= 127, "INT_FAST8_MAX >= 127");
    static_assert(INT_FAST16_MAX >= 32767, "INT_FAST16_MAX >= 32767");
    static_assert(INT_FAST32_MAX >= 2147483647, "INT_FAST32_MAX >= 2147483647");
    static_assert(INT_FAST64_MAX >= 9223372036854775807LL, "INT_FAST64_MAX >= 9223372036854775807LL");

    // UINT_FASTN_MAX
    static_assert(UINT_FAST8_MAX >= 255, "UINT_FAST8_MAX >= 255");
    static_assert(UINT_FAST16_MAX >= 65535, "UINT_FAST16_MAX >= 65535");
    static_assert(UINT_FAST32_MAX >= 4294967295U, "UINT_FAST32_MAX >= 4294967295");
    static_assert(UINT_FAST64_MAX >= 18446744073709551615ULL, "UINT_FAST64_MAX >= 18446744073709551615ULL");

#if 0
    // INTPTR_MIN
    assert(INTPTR_MIN == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::intptr_t>::min());

    // INTPTR_MAX
    assert(INTPTR_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::intptr_t>::max());

    // UINTPTR_MAX
    assert(UINTPTR_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::uintptr_t>::max());

    // INTMAX_MIN
    assert(INTMAX_MIN == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::intmax_t>::min());

    // INTMAX_MAX
    assert(INTMAX_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::intmax_t>::max());

    // UINTMAX_MAX
    assert(UINTMAX_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::uintmax_t>::max());

    // PTRDIFF_MIN
    assert(PTRDIFF_MIN == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::ptrdiff_t>::min());

    // PTRDIFF_MAX
    assert(PTRDIFF_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::ptrdiff_t>::max());

    // SIG_ATOMIC_MIN
    // assert(SIG_ATOMIC_MIN == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::sig_atomic_t>::min());

    // SIG_ATOMIC_MAX
    // assert(SIG_ATOMIC_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::sig_atomic_t>::max());

    // SIZE_MAX
    assert(SIZE_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::size_t>::max());

    // WCHAR_MIN
    // assert(WCHAR_MIN == cuda_for_dali::std::numeric_limits<wchar_t>::min());

    // WCHAR_MAX
    // assert(WCHAR_MAX == cuda_for_dali::std::numeric_limits<wchar_t>::max());

    // WINT_MIN
    // assert(WINT_MIN == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::wint_t>::min());

    // WINT_MAX
    // assert(WINT_MAX == cuda_for_dali::std::numeric_limits<cuda_for_dali::std::wint_t>::max());
#endif

#ifndef INT8_C
#error INT8_C not defined
#endif

#ifndef INT16_C
#error INT16_C not defined
#endif

#ifndef INT32_C
#error INT32_C not defined
#endif

#ifndef INT64_C
#error INT64_C not defined
#endif

#ifndef UINT8_C
#error UINT8_C not defined
#endif

#ifndef UINT16_C
#error UINT16_C not defined
#endif

#ifndef UINT32_C
#error UINT32_C not defined
#endif

#ifndef UINT64_C
#error UINT64_C not defined
#endif

#ifndef INTMAX_C
#error INTMAX_C not defined
#endif

#ifndef UINTMAX_C
#error UINTMAX_C not defined
#endif

  return 0;
}
