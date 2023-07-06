//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... Tuples> tuple<CTypes...> tuple_cat(Tuples&&... tpls);

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/utility>
// cuda_for_dali::std::string not supported
//#include <cuda_for_dali/std/array>
// cuda_for_dali::std::array not supported
//#include <cuda_for_dali/std/string>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

int main(int, char**)
{
    {
        cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat();
        unused(t); // Prevent unused warning
    }
    {
        cuda_for_dali::std::tuple<> t1;
        cuda_for_dali::std::tuple<> t2 = cuda_for_dali::std::tuple_cat(t1);
        unused(t2); // Prevent unused warning
    }
    {
        cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::tuple<>());
        unused(t); // Prevent unused warning
    }
    // cuda_for_dali::std::array not supported
    /*
    {
        cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::array<int, 0>());
        unused(t); // Prevent unused warning
    }
    */
    {
        cuda_for_dali::std::tuple<int> t1(1);
        cuda_for_dali::std::tuple<int> t = cuda_for_dali::std::tuple_cat(t1);
        assert(cuda_for_dali::std::get<0>(t) == 1);
    }

#if TEST_STD_VER > 11
    {
        constexpr cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat();
        unused(t); // Prevent unused warning
    }
    {
        constexpr cuda_for_dali::std::tuple<> t1;
        constexpr cuda_for_dali::std::tuple<> t2 = cuda_for_dali::std::tuple_cat(t1);
        unused(t2); // Prevent unused warning
    }
    {
        constexpr cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::tuple<>());
        unused(t); // Prevent unused warning
    }
    // cuda_for_dali::std::array not supported
    /*
    {
        constexpr cuda_for_dali::std::tuple<> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::array<int, 0>());
        unused(t); // Prevent unused warning
    }
    */
    {
        constexpr cuda_for_dali::std::tuple<int> t1(1);
        constexpr cuda_for_dali::std::tuple<int> t = cuda_for_dali::std::tuple_cat(t1);
        static_assert(cuda_for_dali::std::get<0>(t) == 1, "");
    }
    {
        constexpr cuda_for_dali::std::tuple<int> t1(1);
        constexpr cuda_for_dali::std::tuple<int, int> t = cuda_for_dali::std::tuple_cat(t1, t1);
        static_assert(cuda_for_dali::std::get<0>(t) == 1, "");
        static_assert(cuda_for_dali::std::get<1>(t) == 1, "");
    }
#endif
    {
        cuda_for_dali::std::tuple<int, MoveOnly> t =
                                cuda_for_dali::std::tuple_cat(cuda_for_dali::std::tuple<int, MoveOnly>(1, 2));
        assert(cuda_for_dali::std::get<0>(t) == 1);
        assert(cuda_for_dali::std::get<1>(t) == 2);
    }
    // cuda_for_dali::std::array not supported
    /*
    {
        cuda_for_dali::std::tuple<int, int, int> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::array<int, 3>());
        assert(cuda_for_dali::std::get<0>(t) == 0);
        assert(cuda_for_dali::std::get<1>(t) == 0);
        assert(cuda_for_dali::std::get<2>(t) == 0);
    }
    */
    {
        cuda_for_dali::std::tuple<int, MoveOnly> t = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::pair<int, MoveOnly>(2, 1));
        assert(cuda_for_dali::std::get<0>(t) == 2);
        assert(cuda_for_dali::std::get<1>(t) == 1);
    }

    {
        cuda_for_dali::std::tuple<> t1;
        cuda_for_dali::std::tuple<> t2;
        cuda_for_dali::std::tuple<> t3 = cuda_for_dali::std::tuple_cat(t1, t2);
        unused(t3); // Prevent unused warning
    }
    {
        cuda_for_dali::std::tuple<> t1;
        cuda_for_dali::std::tuple<int> t2(2);
        cuda_for_dali::std::tuple<int> t3 = cuda_for_dali::std::tuple_cat(t1, t2);
        assert(cuda_for_dali::std::get<0>(t3) == 2);
    }
    {
        cuda_for_dali::std::tuple<> t1;
        cuda_for_dali::std::tuple<int> t2(2);
        cuda_for_dali::std::tuple<int> t3 = cuda_for_dali::std::tuple_cat(t2, t1);
        assert(cuda_for_dali::std::get<0>(t3) == 2);
    }
    {
        cuda_for_dali::std::tuple<int*> t1;
        cuda_for_dali::std::tuple<int> t2(2);
        cuda_for_dali::std::tuple<int*, int> t3 = cuda_for_dali::std::tuple_cat(t1, t2);
        assert(cuda_for_dali::std::get<0>(t3) == nullptr);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
    }
    {
        cuda_for_dali::std::tuple<int*> t1;
        cuda_for_dali::std::tuple<int> t2(2);
        cuda_for_dali::std::tuple<int, int*> t3 = cuda_for_dali::std::tuple_cat(t2, t1);
        assert(cuda_for_dali::std::get<0>(t3) == 2);
        assert(cuda_for_dali::std::get<1>(t3) == nullptr);
    }
    {
        cuda_for_dali::std::tuple<int*> t1;
        cuda_for_dali::std::tuple<int, double> t2(2, 3.5);
        cuda_for_dali::std::tuple<int*, int, double> t3 = cuda_for_dali::std::tuple_cat(t1, t2);
        assert(cuda_for_dali::std::get<0>(t3) == nullptr);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == 3.5);
    }
    {
        cuda_for_dali::std::tuple<int*> t1;
        cuda_for_dali::std::tuple<int, double> t2(2, 3.5);
        cuda_for_dali::std::tuple<int, double, int*> t3 = cuda_for_dali::std::tuple_cat(t2, t1);
        assert(cuda_for_dali::std::get<0>(t3) == 2);
        assert(cuda_for_dali::std::get<1>(t3) == 3.5);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
    }
    {
        cuda_for_dali::std::tuple<int*, MoveOnly> t1(nullptr, 1);
        cuda_for_dali::std::tuple<int, double> t2(2, 3.5);
        cuda_for_dali::std::tuple<int*, MoveOnly, int, double> t3 =
                                              cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t1), t2);
        assert(cuda_for_dali::std::get<0>(t3) == nullptr);
        assert(cuda_for_dali::std::get<1>(t3) == 1);
        assert(cuda_for_dali::std::get<2>(t3) == 2);
        assert(cuda_for_dali::std::get<3>(t3) == 3.5);
    }
    {
        cuda_for_dali::std::tuple<int*, MoveOnly> t1(nullptr, 1);
        cuda_for_dali::std::tuple<int, double> t2(2, 3.5);
        cuda_for_dali::std::tuple<int, double, int*, MoveOnly> t3 =
                                              cuda_for_dali::std::tuple_cat(t2, cuda_for_dali::std::move(t1));
        assert(cuda_for_dali::std::get<0>(t3) == 2);
        assert(cuda_for_dali::std::get<1>(t3) == 3.5);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 1);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cuda_for_dali::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t1), cuda_for_dali::std::move(t2));
        assert(cuda_for_dali::std::get<0>(t3) == 1);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 4);
    }

    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cuda_for_dali::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cuda_for_dali::std::tuple_cat(cuda_for_dali::std::tuple<>(),
                                                  cuda_for_dali::std::move(t1),
                                                  cuda_for_dali::std::move(t2));
        assert(cuda_for_dali::std::get<0>(t3) == 1);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 4);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cuda_for_dali::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t1),
                                                  cuda_for_dali::std::tuple<>(),
                                                  cuda_for_dali::std::move(t2));
        assert(cuda_for_dali::std::get<0>(t3) == 1);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 4);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cuda_for_dali::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly> t3 =
                                   cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t1),
                                                  cuda_for_dali::std::move(t2),
                                                  cuda_for_dali::std::tuple<>());
        assert(cuda_for_dali::std::get<0>(t3) == 1);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 4);
    }
    {
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly> t1(1, 2);
        cuda_for_dali::std::tuple<int*, MoveOnly> t2(nullptr, 4);
        cuda_for_dali::std::tuple<MoveOnly, MoveOnly, int*, MoveOnly, int> t3 =
                                   cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t1),
                                                  cuda_for_dali::std::move(t2),
                                                  cuda_for_dali::std::tuple<int>(5));
        assert(cuda_for_dali::std::get<0>(t3) == 1);
        assert(cuda_for_dali::std::get<1>(t3) == 2);
        assert(cuda_for_dali::std::get<2>(t3) == nullptr);
        assert(cuda_for_dali::std::get<3>(t3) == 4);
        assert(cuda_for_dali::std::get<4>(t3) == 5);
    }
    {
        // See bug #19616.
        auto t1 = cuda_for_dali::std::tuple_cat(
            cuda_for_dali::std::make_tuple(cuda_for_dali::std::make_tuple(1)),
            cuda_for_dali::std::make_tuple()
        );
        assert(t1 == cuda_for_dali::std::make_tuple(cuda_for_dali::std::make_tuple(1)));

        auto t2 = cuda_for_dali::std::tuple_cat(
            cuda_for_dali::std::make_tuple(cuda_for_dali::std::make_tuple(1)),
            cuda_for_dali::std::make_tuple(cuda_for_dali::std::make_tuple(2))
        );
        assert(t2 == cuda_for_dali::std::make_tuple(cuda_for_dali::std::make_tuple(1), cuda_for_dali::std::make_tuple(2)));
    }
    {
        int x = 101;
        cuda_for_dali::std::tuple<int, const int, int&, const int&, int&&> t(42, 101, x, x, cuda_for_dali::std::move(x));
        const auto& ct = t;
        cuda_for_dali::std::tuple<int, const int, int&, const int&> t2(42, 101, x, x);
        const auto& ct2 = t2;

        auto r = cuda_for_dali::std::tuple_cat(cuda_for_dali::std::move(t), cuda_for_dali::std::move(ct), t2, ct2);

        ASSERT_SAME_TYPE(decltype(r), cuda_for_dali::std::tuple<
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&, int&&,
            int, const int, int&, const int&,
            int, const int, int&, const int&>);
        unused(r);
    }
  return 0;
}
