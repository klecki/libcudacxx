//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "DefaultOnly.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

struct NoDefault {
    NoDefault() = delete;
    __host__ __device__ explicit NoDefault(int) { }
};

struct NoExceptDefault {
    NoExceptDefault() noexcept = default;
};

struct ThrowingDefault {
    __host__ __device__ ThrowingDefault() { }
};

struct IllFormedDefault {
    __host__ __device__ IllFormedDefault(int x) : value(x) {}
    template <bool Pred = false>
    __host__ __device__ constexpr IllFormedDefault() {
        static_assert(Pred,
            "The default constructor should not be instantiated");
    }
    int value;
};

int main(int, char**)
{
    {
        cuda_for_dali::std::tuple<> t;
        unused(t);
    }
    {
        cuda_for_dali::std::tuple<int> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
    }
    {
        cuda_for_dali::std::tuple<int, char*> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == nullptr);
    }
    // cuda_for_dali::std::string not supported
    /*
    {
        cuda_for_dali::std::tuple<int, char*, cuda_for_dali::std::string> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == nullptr);
        assert(cuda_for_dali::std::get<2>(t) == "");
    }
    {
        cuda_for_dali::std::tuple<int, char*, cuda_for_dali::std::string, DefaultOnly> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == nullptr);
        assert(cuda_for_dali::std::get<2>(t) == "");
        assert(cuda_for_dali::std::get<3>(t) == DefaultOnly());
    }
    */
    {
        // See bug #21157.
        static_assert(!cuda_for_dali::std::is_default_constructible<cuda_for_dali::std::tuple<NoDefault>>(), "");
        static_assert(!cuda_for_dali::std::is_default_constructible<cuda_for_dali::std::tuple<DefaultOnly, NoDefault>>(), "");
        static_assert(!cuda_for_dali::std::is_default_constructible<cuda_for_dali::std::tuple<NoDefault, DefaultOnly, NoDefault>>(), "");
    }
    {
        static_assert(noexcept(cuda_for_dali::std::tuple<NoExceptDefault>()), "");
        static_assert(noexcept(cuda_for_dali::std::tuple<NoExceptDefault, NoExceptDefault>()), "");

        static_assert(!noexcept(cuda_for_dali::std::tuple<ThrowingDefault, NoExceptDefault>()), "");
        static_assert(!noexcept(cuda_for_dali::std::tuple<NoExceptDefault, ThrowingDefault>()), "");
        static_assert(!noexcept(cuda_for_dali::std::tuple<ThrowingDefault, ThrowingDefault>()), "");
    }
    {
        constexpr cuda_for_dali::std::tuple<> t;
        unused(t);
    }
    {
        constexpr cuda_for_dali::std::tuple<int> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
    }
    {
        constexpr cuda_for_dali::std::tuple<int, char*> t;
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == nullptr);
    }
    {
    // Check that the SFINAE on the default constructor is not evaluated when
    // it isn't needed. If the default constructor is evaluated then this test
    // should fail to compile.
        IllFormedDefault v(0);
        cuda_for_dali::std::tuple<IllFormedDefault> t(v);
        unused(t);
    }
    {
        struct Base { };
        struct Derived : Base { protected: Derived() = default; };
        static_assert(!cuda_for_dali::std::is_default_constructible<cuda_for_dali::std::tuple<Derived, int> >::value, "");
    }

    return 0;
}
