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
    static_assert ( !is_transparent<cuda_for_dali::std::plus<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::plus<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::plus<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::plus<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::minus<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::minus<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::minus<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::minus<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::multiplies<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::multiplies<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::multiplies<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::multiplies<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::divides<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::divides<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::divides<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::divides<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::modulus<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::modulus<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::modulus<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::modulus<>>::value, "" );

    static_assert ( !is_transparent<cuda_for_dali::std::negate<int>>::value, "" );
    // static_assert ( !is_transparent<cuda_for_dali::std::negate<cuda_for_dali::std::string>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::negate<void>>::value, "" );
    static_assert (  is_transparent<cuda_for_dali::std::negate<>>::value, "" );

    return 0;
}
