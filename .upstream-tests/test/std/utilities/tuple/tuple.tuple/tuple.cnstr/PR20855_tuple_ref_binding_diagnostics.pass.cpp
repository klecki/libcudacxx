// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// See llvm.org/PR20855

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>
#include "test_macros.h"

#if TEST_HAS_BUILTIN_IDENTIFIER(__reference_binds_to_temporary)
# define ASSERT_REFERENCE_BINDS_TEMPORARY(...) static_assert(__reference_binds_to_temporary(__VA_ARGS__), "")
# define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(!__reference_binds_to_temporary(__VA_ARGS__), "")
#else
# define ASSERT_REFERENCE_BINDS_TEMPORARY(...) static_assert(true, "")
# define ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(...) static_assert(true, "")
#endif

template <class Tp>
struct ConvertsTo {
  using RawTp = typename cuda_for_dali::std::remove_cv< typename cuda_for_dali::std::remove_reference<Tp>::type>::type;

  __host__ __device__ operator Tp() const {
    return static_cast<Tp>(value);
  }

  mutable RawTp value;
};

struct Base {};
struct Derived : Base {};


static_assert(cuda_for_dali::std::is_same<decltype("abc"), decltype(("abc"))>::value, "");
// cuda_for_dali::std::string not supported
/*
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda_for_dali::std::string const&, decltype("abc"));
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda_for_dali::std::string const&, decltype(("abc")));
ASSERT_REFERENCE_BINDS_TEMPORARY(cuda_for_dali::std::string const&, const char*&&);
*/
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(int&, const ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(const int&, ConvertsTo<int&>&);
ASSERT_NOT_REFERENCE_BINDS_TEMPORARY(Base&, Derived&);


static_assert(cuda_for_dali::std::is_constructible<int&, cuda_for_dali::std::reference_wrapper<int>>::value, "");
static_assert(cuda_for_dali::std::is_constructible<int const&, cuda_for_dali::std::reference_wrapper<int>>::value, "");

template <class T> struct CannotDeduce {
  using type = T;
};

template <class ...Args>
__host__ __device__ void F(typename CannotDeduce<cuda_for_dali::std::tuple<Args...>>::type const&) {}

__host__ __device__ void compile_tests() {
  {
    F<int, int const&>(cuda_for_dali::std::make_tuple(42, 42));
  }
  {
    F<int, int const&>(cuda_for_dali::std::make_tuple<const int&, const int&>(42, 42));
    cuda_for_dali::std::tuple<int, int const&> t(cuda_for_dali::std::make_tuple<const int&, const int&>(42, 42));
  }
  // cuda_for_dali::std::string not supported
  /*
  {
    auto fn = &F<int, cuda_for_dali::std::string const&>;
    fn(cuda_for_dali::std::tuple<int, cuda_for_dali::std::string const&>(42, cuda_for_dali::std::string("a")));
    fn(cuda_for_dali::std::make_tuple(42, cuda_for_dali::std::string("a")));
  }
  */
  {
    Derived d;
    cuda_for_dali::std::tuple<Base&, Base const&> t(d, d);
  }
  {
    ConvertsTo<int&> ct;
    cuda_for_dali::std::tuple<int, int&> t(42, ct);
  }
}

__host__ __device__ void allocator_tests() {
    // cuda_for_dali::std::allocator not supported
    //cuda_for_dali::std::allocator<void> alloc;
    int x = 42;
    {
        cuda_for_dali::std::tuple<int&> t(cuda_for_dali::std::ref(x));
        assert(&cuda_for_dali::std::get<0>(t) == &x);
        // cuda_for_dali::std::allocator not supported
        /*
        cuda_for_dali::std::tuple<int&> t1(cuda_for_dali::std::allocator_arg, alloc, cuda_for_dali::std::ref(x));
        assert(&cuda_for_dali::std::get<0>(t1) == &x);
        */
    }
    {
        auto r = cuda_for_dali::std::ref(x);
        auto const& cr = r;
        cuda_for_dali::std::tuple<int&> t(r);
        assert(&cuda_for_dali::std::get<0>(t) == &x);
        cuda_for_dali::std::tuple<int&> t1(cr);
        assert(&cuda_for_dali::std::get<0>(t1) == &x);
        // cuda_for_dali::std::allocator not supported
        /*
        cuda_for_dali::std::tuple<int&> t2(cuda_for_dali::std::allocator_arg, alloc, r);
        assert(&cuda_for_dali::std::get<0>(t2) == &x);
        cuda_for_dali::std::tuple<int&> t3(cuda_for_dali::std::allocator_arg, alloc, cr);
        assert(&cuda_for_dali::std::get<0>(t3) == &x);
        */
    }
    {
        cuda_for_dali::std::tuple<int const&> t(cuda_for_dali::std::ref(x));
        assert(&cuda_for_dali::std::get<0>(t) == &x);
        cuda_for_dali::std::tuple<int const&> t2(cuda_for_dali::std::cref(x));
        assert(&cuda_for_dali::std::get<0>(t2) == &x);
        // cuda_for_dali::std::allocator not supported
        /*
        cuda_for_dali::std::tuple<int const&> t3(cuda_for_dali::std::allocator_arg, alloc, cuda_for_dali::std::ref(x));
        assert(&cuda_for_dali::std::get<0>(t3) == &x);
        cuda_for_dali::std::tuple<int const&> t4(cuda_for_dali::std::allocator_arg, alloc, cuda_for_dali::std::cref(x));
        assert(&cuda_for_dali::std::get<0>(t4) == &x);
        */
    }
    {
        auto r = cuda_for_dali::std::ref(x);
        auto cr = cuda_for_dali::std::cref(x);
        cuda_for_dali::std::tuple<int const&> t(r);
        assert(&cuda_for_dali::std::get<0>(t) == &x);
        cuda_for_dali::std::tuple<int const&> t2(cr);
        assert(&cuda_for_dali::std::get<0>(t2) == &x);
        // cuda_for_dali::std::allocator not supported
        /*
        cuda_for_dali::std::tuple<int const&> t3(cuda_for_dali::std::allocator_arg, alloc, r);
        assert(&cuda_for_dali::std::get<0>(t3) == &x);
        cuda_for_dali::std::tuple<int const&> t4(cuda_for_dali::std::allocator_arg, alloc, cr);
        assert(&cuda_for_dali::std::get<0>(t4) == &x);
        */
    }
}


int main(int, char**) {
  compile_tests();
  allocator_tests();

  return 0;
}
