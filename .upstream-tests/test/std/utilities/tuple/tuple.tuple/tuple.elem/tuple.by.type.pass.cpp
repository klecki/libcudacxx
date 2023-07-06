//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// UNSUPPORTED: c++98, c++03, c++11

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::unique_ptr not supported
//#include <cuda_for_dali/std/memory>
// cuda_for_dali::std::string not supported
//#include <cuda_for_dali/std/string>
// cuda_for_dali::std::complex not supported
//#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/type_traits>

#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    // cuda_for_dali::std::complex not supported
    // cuda_for_dali::std::string not supported
    /*
    typedef cuda_for_dali::std::complex<float> cf;
    {
    auto t1 = cuda_for_dali::std::tuple<int, cuda_for_dali::std::string, cf> { 42, "Hi", { 1,2 }};
    assert ( cuda_for_dali::std::get<int>(t1) == 42 ); // find at the beginning
    assert ( cuda_for_dali::std::get<cuda_for_dali::std::string>(t1) == "Hi" ); // find in the middle
    assert ( cuda_for_dali::std::get<cf>(t1).real() == 1 ); // find at the end
    assert ( cuda_for_dali::std::get<cf>(t1).imag() == 2 );
    }

    {
    auto t2 = cuda_for_dali::std::tuple<int, cuda_for_dali::std::string, int, cf> { 42, "Hi", 23, { 1,2 }};
//  get<int> would fail!
    assert ( cuda_for_dali::std::get<cuda_for_dali::std::string>(t2) == "Hi" );
    assert (( cuda_for_dali::std::get<cf>(t2) == cf{ 1,2 } ));
    }
    */
    {
    constexpr cuda_for_dali::std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    static_assert ( cuda_for_dali::std::get<int>(p5) == 1, "" );
    static_assert ( cuda_for_dali::std::get<const int>(p5) == 2, "" );
    }

    {
    const cuda_for_dali::std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    const int &i1 = cuda_for_dali::std::get<int>(p5);
    const int &i2 = cuda_for_dali::std::get<const int>(p5);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    // cuda_for_dali::std::unique_ptr not supported
    /*
    {
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    cuda_for_dali::std::tuple<upint> t(upint(new int(4)));
    upint p = cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t)); // get rvalue
    assert(*p == 4);
    assert(cuda_for_dali::std::get<upint>(t) == nullptr); // has been moved from
    }

    {
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    const cuda_for_dali::std::tuple<upint> t(upint(new int(4)));
    const upint&& p = cuda_for_dali::std::get<upint>(cuda_for_dali::std::move(t)); // get const rvalue
    assert(*p == 4);
    assert(cuda_for_dali::std::get<upint>(t) != nullptr);
    }
    */

    {
    int x = 42;
    int y = 43;
    cuda_for_dali::std::tuple<int&, int const&> const t(x, y);
    static_assert(cuda_for_dali::std::is_same<int&, decltype(cuda_for_dali::std::get<int&>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int&>(cuda_for_dali::std::move(t))), "");
    static_assert(cuda_for_dali::std::is_same<int const&, decltype(cuda_for_dali::std::get<int const&>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int const&>(cuda_for_dali::std::move(t))), "");
    }

    {
    int x = 42;
    int y = 43;
    cuda_for_dali::std::tuple<int&&, int const&&> const t(cuda_for_dali::std::move(x), cuda_for_dali::std::move(y));
    static_assert(cuda_for_dali::std::is_same<int&&, decltype(cuda_for_dali::std::get<int&&>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int&&>(cuda_for_dali::std::move(t))), "");
    static_assert(cuda_for_dali::std::is_same<int const&&, decltype(cuda_for_dali::std::get<int const&&>(cuda_for_dali::std::move(t)))>::value, "");
    static_assert(noexcept(cuda_for_dali::std::get<int const&&>(cuda_for_dali::std::move(t))), "");
    }

    {
    constexpr const cuda_for_dali::std::tuple<int, const int, double, double> t { 1, 2, 3.4, 5.6 };
    static_assert(cuda_for_dali::std::get<int>(cuda_for_dali::std::move(t)) == 1, "");
    static_assert(cuda_for_dali::std::get<const int>(cuda_for_dali::std::move(t)) == 2, "");
    }

  return 0;
}
