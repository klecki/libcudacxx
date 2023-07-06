//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
#include <cuda_for_dali/std/functional>
// #include <cuda_for_dali/std/string>

template <class T>
struct is_transparent
{
private:
    struct two {char lx; char lxx;};
    template <class U> __host__ __device__ static two test(...);
    template <class U> __host__ __device__ static char test(typename U::is_transparent* = 0);
public:
    static const bool value = sizeof(test<T>(0)) == 1;
};


int main(int, char**)
{
    static_assert ( !is_transparent<cuda_for_dali::std::less<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::less<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::less<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::less<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::less_equal<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::less_equal<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::less_equal<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::less_equal<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::equal_to<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::equal_to<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::equal_to<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::equal_to<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::not_equal_to<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::not_equal_to<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::not_equal_to<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::not_equal_to<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::greater<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::greater<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::greater<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::greater<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::greater_equal<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::greater_equal<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::greater_equal<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::greater_equal<>>::value, "" );

    return 0;
}
