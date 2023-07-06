//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

struct Explicit {
  int value;
  __host__ __device__ explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  __host__ __device__ Implicit(int x) : value(x) {}
};

struct ExplicitTwo {
    __host__ __device__ ExplicitTwo() {}
    __host__ __device__ ExplicitTwo(ExplicitTwo const&) {}
    __host__ __device__ ExplicitTwo(ExplicitTwo &&) {}

    template <class T, class = typename cuda_for_dali::std::enable_if<!cuda_for_dali::std::is_same<T, ExplicitTwo>::value>::type>
    __host__ __device__ explicit ExplicitTwo(T) {}
};

struct B
{
    int id_;

    __host__ __device__ explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    __host__ __device__ explicit D(int i) : B(i) {}
};

#if TEST_STD_VER > 11

struct A
{
    int id_;

    __host__ __device__ constexpr A(int i) : id_(i) {}
    __host__ __device__ friend constexpr bool operator==(const A& x, const A& y) {return x.id_ == y.id_;}
};

struct C
{
    int id_;

    __host__ __device__ constexpr explicit C(int i) : id_(i) {}
    __host__ __device__ friend constexpr bool operator==(const C& x, const C& y) {return x.id_ == y.id_;}
};

#endif

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::tuple<long> T0;
        typedef cuda_for_dali::std::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = t0;
        assert(cuda_for_dali::std::get<0>(t1) == 2);
    }
#if TEST_STD_VER > 11
    {
        typedef cuda_for_dali::std::tuple<int> T0;
        typedef cuda_for_dali::std::tuple<A> T1;
        constexpr T0 t0(2);
        constexpr T1 t1 = t0;
        static_assert(cuda_for_dali::std::get<0>(t1) == 2, "");
    }
    {
        typedef cuda_for_dali::std::tuple<int> T0;
        typedef cuda_for_dali::std::tuple<C> T1;
        constexpr T0 t0(2);
        constexpr T1 t1{t0};
        static_assert(cuda_for_dali::std::get<0>(t1) == C(2), "");
    }
#endif
    {
        typedef cuda_for_dali::std::tuple<long, char> T0;
        typedef cuda_for_dali::std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = t0;
        assert(cuda_for_dali::std::get<0>(t1) == 2);
        assert(cuda_for_dali::std::get<1>(t1) == int('a'));
    }
    {
        typedef cuda_for_dali::std::tuple<long, char, D> T0;
        typedef cuda_for_dali::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = t0;
        assert(cuda_for_dali::std::get<0>(t1) == 2);
        assert(cuda_for_dali::std::get<1>(t1) == int('a'));
        assert(cuda_for_dali::std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        typedef cuda_for_dali::std::tuple<long, char, D&> T0;
        typedef cuda_for_dali::std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        assert(cuda_for_dali::std::get<0>(t1) == 2);
        assert(cuda_for_dali::std::get<1>(t1) == int('a'));
        assert(cuda_for_dali::std::get<2>(t1).id_ == 2);
    }
    {
        typedef cuda_for_dali::std::tuple<long, char, int> T0;
        typedef cuda_for_dali::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', 3);
        T1 t1(t0);
        assert(cuda_for_dali::std::get<0>(t1) == 2);
        assert(cuda_for_dali::std::get<1>(t1) == int('a'));
        assert(cuda_for_dali::std::get<2>(t1).id_ == 3);
    }
    {
        const cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<Explicit> t2(t1);
        assert(cuda_for_dali::std::get<0>(t2).value == 42);
    }
    {
        const cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<Implicit> t2 = t1;
        assert(cuda_for_dali::std::get<0>(t2).value == 42);
    }
    {
        static_assert(cuda_for_dali::std::is_convertible<ExplicitTwo&&, ExplicitTwo>::value, "");
        static_assert(cuda_for_dali::std::is_convertible<cuda_for_dali::std::tuple<ExplicitTwo&&>&&, const cuda_for_dali::std::tuple<ExplicitTwo>&>::value, "");

        ExplicitTwo e;
        cuda_for_dali::std::tuple<ExplicitTwo> t = cuda_for_dali::std::tuple<ExplicitTwo&&>(cuda_for_dali::std::move(e));
        ((void)t);
    }
  return 0;
}
