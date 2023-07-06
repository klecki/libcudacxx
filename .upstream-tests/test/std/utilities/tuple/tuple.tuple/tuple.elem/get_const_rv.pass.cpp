//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::string not supported
//#include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef cuda_for_dali::std::tuple<int> T;
    const T t(3);
    static_assert(cuda_for_dali::std::is_same<const int&&, decltype(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t))), "");
    const int&& i = cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t));
    assert(i == 3);
    }

    // cuda_for_dali::std::string not supported
    /*
    {
    typedef cuda_for_dali::std::tuple<cuda_for_dali::std::string, int> T;
    const T t("high", 5);
    static_assert(cuda_for_dali::std::is_same<const cuda_for_dali::std::string&&, decltype(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t))), "");
    static_assert(cuda_for_dali::std::is_same<const int&&, decltype(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(t))), "");
    const cuda_for_dali::std::string&& s = cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t));
    const int&& i = cuda_for_dali::std::get<1>(cuda_for_dali::std::move(t));
    assert(s == "high");
    assert(i == 5);
    }
    */

    {
    int x = 42;
    int const y = 43;
    cuda_for_dali::std::tuple<int&, int const&> const p(x, y);
    static_assert(cuda_for_dali::std::is_same<int&, decltype(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(p))), "");
    static_assert(cuda_for_dali::std::is_same<int const&, decltype(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(p))), "");
    }

    {
    int x = 42;
    int const y = 43;
    cuda_for_dali::std::tuple<int&&, int const&&> const p(cuda_for_dali::std::move(x), cuda_for_dali::std::move(y));
    static_assert(cuda_for_dali::std::is_same<int&&, decltype(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(p))), "");
    static_assert(cuda_for_dali::std::is_same<int const&&, decltype(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(p))), "");
    }

#if TEST_STD_VER > 11
    {
    typedef cuda_for_dali::std::tuple<double, int> T;
    constexpr const T t(2.718, 5);
    static_assert(cuda_for_dali::std::get<0>(cuda_for_dali::std::move(t)) == 2.718, "");
    static_assert(cuda_for_dali::std::get<1>(cuda_for_dali::std::move(t)) == 5, "");
    }
#endif

  return 0;
}
