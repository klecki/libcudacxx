//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// UNSUPPORTED: msvc

#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::string not supported
// #include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/type_traits>
// cuda/std/complex not supported
// #include <cuda_for_dali/std/complex>
// cuda/std/memory not supported
// #include <cuda_for_dali/std/memory>

#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // cuda/std/complex not supported
    /*
    typedef cuda_for_dali::std::complex<float> cf;
    {
    auto t1 = cuda_for_dali::std::make_pair<int, cf> ( 42, { 1,2 } );
    assert ( cuda_for_dali::std::get<int>(t1) == 42 );
    assert ( cuda_for_dali::std::get<cf>(t1).real() == 1 );
    assert ( cuda_for_dali::std::get<cf>(t1).imag() == 2 );
    }
    */
    {
    const cuda_for_dali::std::pair<int, const int> p1 { 1, 2 };
    const int &i1 = cuda_for_dali::std::get<int>(p1);
    const int &i2 = cuda_for_dali::std::get<const int>(p1);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    // cuda/std/memory not supported
    /*
    {
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    cuda_for_dali::std::pair<upint, int> t(upint(new int(4)), 42);
    upint p = cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t)); // get rvalue
    assert(*p == 4);
    assert(cuda_for_dali::std::get<upint>(t) == nullptr); // has been moved from
    }

    {
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    const cuda_for_dali::std::pair<upint, int> t(upint(new int(4)), 42);
    static_assert(cuda_for_dali::std::is_same<const upint&&, decltype(cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t))), "");
    static_assert(cuda_for_dali::std::is_same<const int&&, decltype(cuda_for_dali::std::get<int>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int>(cuda_for_dali::std::move(t))), "");
    auto&& p = cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t)); // get const rvalue
    auto&& i = cuda_for_dali::std::get<int>(cuda_for_dali::std::move(t)); // get const rvalue
    assert(*p == 4);
    assert(i == 42);
    assert(cuda_for_dali::std::get<upint>(t) != nullptr);
    }
    */

    {
    int x = 42;
    int const y = 43;
    cuda_for_dali::std::pair<int&, int const&> const p(x, y);
    static_assert(cuda_for_dali::std::is_same<int&, decltype(cuda_for_dali::std::get<int&>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int&>(cuda_for_dali::std::move(p))), "");
    static_assert(cuda_for_dali::std::is_same<int const&, decltype(cuda_for_dali::std::get<int const&>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int const&>(cuda_for_dali::std::move(p))), "");
    }

    {
    int x = 42;
    int const y = 43;
    cuda_for_dali::std::pair<int&&, int const&&> const p(cuda_for_dali::std::move(x), cuda_for_dali::std::move(y));
    static_assert(cuda_for_dali::std::is_same<int&&, decltype(cuda_for_dali::std::get<int&&>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int&&>(cuda_for_dali::std::move(p))), "");
    static_assert(cuda_for_dali::std::is_same<int const&&, decltype(cuda_for_dali::std::get<int const&&>(cuda_for_dali::std::move(p)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int const&&>(cuda_for_dali::std::move(p))), "");
    }

    {
    constexpr const cuda_for_dali::std::pair<int, const int> p { 1, 2 };
    static_assert(cuda_for_dali::std::get<int>(cuda_for_dali::std::move(p)) == 1, "");
    static_assert(cuda_for_dali::std::get<const int>(cuda_for_dali::std::move(p)) == 2, "");
    }

  return 0;
}
