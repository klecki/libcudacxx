//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "test_convertible.h"
#include "MoveOnly.h"

#if TEST_STD_VER > 11

struct Empty {};
struct A
{
    int id_;
    __host__ __device__ explicit constexpr A(int i) : id_(i) {}
};

#endif

struct NoDefault { NoDefault() = delete; };

// Make sure the _Up... constructor SFINAEs out when the types that
// are not explicitly initialized are not all default constructible.
// Otherwise, cuda_for_dali::std::is_constructible would return true but instantiating
// the constructor would fail.
__host__ __device__ void test_default_constructible_extension_sfinae()
{
    {
        typedef cuda_for_dali::std::tuple<MoveOnly, NoDefault> Tuple;

        static_assert(!cuda_for_dali::std::is_constructible<
            Tuple,
            MoveOnly
        >::value, "");

        static_assert(cuda_for_dali::std::is_constructible<
            Tuple,
            MoveOnly, NoDefault
        >::value, "");
    }
    {
        typedef cuda_for_dali::std::tuple<MoveOnly, MoveOnly, NoDefault> Tuple;

        static_assert(!cuda_for_dali::std::is_constructible<
            Tuple,
            MoveOnly, MoveOnly
        >::value, "");

        static_assert(cuda_for_dali::std::is_constructible<
            Tuple,
            MoveOnly, MoveOnly, NoDefault
        >::value, "");
    }
    {
        // Same idea as above but with a nested tuple type.
        typedef cuda_for_dali::std::tuple<MoveOnly, NoDefault> Tuple;
        typedef cuda_for_dali::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(!cuda_for_dali::std::is_constructible<
            NestedTuple,
            MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(cuda_for_dali::std::is_constructible<
            NestedTuple,
            MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
    // testing extensions
#ifdef _LIBCUDAFORDALICXX_VERSION
    {
        typedef cuda_for_dali::std::tuple<MoveOnly, int> Tuple;
        typedef cuda_for_dali::std::tuple<MoveOnly, Tuple, MoveOnly, MoveOnly> NestedTuple;

        static_assert(cuda_for_dali::std::is_constructible<
            NestedTuple,
            MoveOnly, MoveOnly, MoveOnly, MoveOnly
        >::value, "");

        static_assert(cuda_for_dali::std::is_constructible<
            NestedTuple,
            MoveOnly, Tuple, MoveOnly, MoveOnly
        >::value, "");
    }
#endif
}

int main(int, char**)
{
    {
        cuda_for_dali::std::tuple<MoveOnly> t(MoveOnly(0));
        assert(cuda_for_dali::std::get<0>(t) == 0);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t(MoveOnly(0), MoveOnly(1));
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == 1);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, MoveOnly> t(MoveOnly(0),
                                                   MoveOnly(1),
                                                   MoveOnly(2));
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == 1);
        assert(cuda_for_dali::std::get<2>(t) == 2);
    }
    // extensions, MSVC issues
#if defined(_LIBCUDAFORDALICXX_VERSION) && !defined(_MSC_VER)
    {
        using E = MoveOnly;
        using Tup = cuda_for_dali::std::tuple<E, E, E>;
        // Test that the reduced arity initialization extension is only
        // allowed on the explicit constructor.
        static_assert(test_convertible<Tup, E, E, E>(), "");

        Tup t(E(0), E(1));
        static_assert(!test_convertible<Tup, E, E>(), "");
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == 1);
        assert(cuda_for_dali::std::get<2>(t) == MoveOnly());

        Tup t2(E(0));
        static_assert(!test_convertible<Tup, E>(), "");
        assert(cuda_for_dali::std::get<0>(t2) == 0);
        assert(cuda_for_dali::std::get<1>(t2) == E());
        assert(cuda_for_dali::std::get<2>(t2) == E());
    }
#endif
#if TEST_STD_VER > 11
    {
        constexpr cuda_for_dali::std::tuple<Empty> t0{Empty()};
        (void)t0;
    }
    {
        constexpr cuda_for_dali::std::tuple<A, A> t(3, 2);
        static_assert(cuda_for_dali::std::get<0>(t).id_ == 3, "");
    }
#endif
    // Check that SFINAE is properly applied with the default reduced arity
    // constructor extensions.
    test_default_constructible_extension_sfinae();

  return 0;
}
