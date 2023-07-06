//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// tuple& operator=(const tuple& u);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct NonAssignable {
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&) = delete;
};
struct CopyAssignable {
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable &&) = delete;
};
static_assert(cuda_for_dali::std::is_copy_assignable<CopyAssignable>::value, "");
struct MoveAssignable {
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&) = default;
};

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::tuple<> T;
        T t0;
        T t;
        t = t0;
        unused(t);
    }
    {
        typedef cuda_for_dali::std::tuple<int> T;
        T t0(2);
        T t;
        t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
    }
    {
        typedef cuda_for_dali::std::tuple<int, char> T;
        T t0(2, 'a');
        T t;
        t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
        assert(cuda_for_dali::std::get<1>(t) == 'a');
    }
    // cuda_for_dali::std::string not supported
    /*
    {
        typedef cuda_for_dali::std::tuple<int, char, cuda_for_dali::std::string> T;
        const T t0(2, 'a', "some text");
        T t;
        t = t0;
        assert(cuda_for_dali::std::get<0>(t) == 2);
        assert(cuda_for_dali::std::get<1>(t) == 'a');
        assert(cuda_for_dali::std::get<2>(t) == "some text");
    }
    */
    {
        // test reference assignment.
        using T = cuda_for_dali::std::tuple<int&, int&&>;
        int x = 42;
        int y = 100;
        int x2 = -1;
        int y2 = 500;
        T t(x, cuda_for_dali::std::move(y));
        T t2(x2, cuda_for_dali::std::move(y2));
        t = t2;
        assert(cuda_for_dali::std::get<0>(t) == x2);
        assert(&cuda_for_dali::std::get<0>(t) == &x);
        assert(cuda_for_dali::std::get<1>(t) == y2);
        assert(&cuda_for_dali::std::get<1>(t) == &y);
    }
    // cuda_for_dali::std::unique_ptr not supported
    /*
    {
        // test that the implicitly generated copy assignment operator
        // is properly deleted
        using T = cuda_for_dali::std::tuple<cuda_for_dali::std::unique_ptr<int>>;
        static_assert(!cuda_for_dali::std::is_copy_assignable<T>::value, "");
    }
    */
    {
        using T = cuda_for_dali::std::tuple<int, NonAssignable>;
        static_assert(!cuda_for_dali::std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cuda_for_dali::std::tuple<int, CopyAssignable>;
        static_assert(cuda_for_dali::std::is_copy_assignable<T>::value, "");
    }
    {
        using T = cuda_for_dali::std::tuple<int, MoveAssignable>;
        static_assert(!cuda_for_dali::std::is_copy_assignable<T>::value, "");
    }

  return 0;
}
