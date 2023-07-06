//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class TupleLike>
//   tuple(TupleLike&&);
// template <class Alloc, class TupleLike>
//   tuple(cuda_for_dali::std::allocator_arg_t, Alloc const&, TupleLike&&);

// Check that the tuple-like ctors are properly disabled when the UTypes...
// constructor should be selected. See PR22806.

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

template <class Tp>
using uncvref_t = typename cuda_for_dali::std::remove_cv<typename cuda_for_dali::std::remove_reference<Tp>::type>::type;

template <class Tuple, class = uncvref_t<Tuple>>
struct IsTuple : cuda_for_dali::std::false_type {};

template <class Tuple, class ...Args>
struct IsTuple<Tuple, cuda_for_dali::std::tuple<Args...>> : cuda_for_dali::std::true_type {};

struct ConstructibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt const&) : state(Copied) {}
  __host__ __device__ ConstructibleFromTupleAndInt(ConstructibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename cuda_for_dali::std::enable_if<IsTuple<Tuple>::value>::type>
  __host__ __device__ explicit ConstructibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  __host__ __device__ explicit ConstructibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConvertibleFromTupleAndInt {
  enum State { FromTuple, FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt const&) : state(Copied) {}
  __host__ __device__ ConvertibleFromTupleAndInt(ConvertibleFromTupleAndInt &&) : state(Moved) {}

  template <class Tuple, class = typename cuda_for_dali::std::enable_if<IsTuple<Tuple>::value>::type>
  __host__ __device__ ConvertibleFromTupleAndInt(Tuple&&) : state(FromTuple) {}

  __host__ __device__ ConvertibleFromTupleAndInt(int) : state(FromInt) {}
};

struct ConstructibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConstructibleFromInt(ConstructibleFromInt const&) : state(Copied) {}
  __host__ __device__ ConstructibleFromInt(ConstructibleFromInt &&) : state(Moved) {}

  __host__ __device__ explicit ConstructibleFromInt(int) : state(FromInt) {}
};

struct ConvertibleFromInt {
  enum State { FromInt, Copied, Moved };
  State state;

  __host__ __device__ ConvertibleFromInt(ConvertibleFromInt const&) : state(Copied) {}
  __host__ __device__ ConvertibleFromInt(ConvertibleFromInt &&) : state(Moved) {}
  __host__ __device__ ConvertibleFromInt(int) : state(FromInt) {}
};

int main(int, char**)
{
    // Test for the creation of dangling references when a tuple is used to
    // store a reference to another tuple as its only element.
    // Ex cuda_for_dali::std::tuple<cuda_for_dali::std::tuple<int>&&>.
    // In this case the constructors 1) 'tuple(UTypes&&...)'
    // and 2) 'tuple(TupleLike&&)' need to be manually disambiguated because
    // when both #1 and #2 participate in partial ordering #2 will always
    // be chosen over #1.
    // See PR22806  and LWG issue #2549 for more information.
    // (https://bugs.llvm.org/show_bug.cgi?id=22806)
    using T = cuda_for_dali::std::tuple<int>;
    // cuda_for_dali::std::allocator not supported
    // cuda_for_dali::std::allocator<int> A;
    { // rvalue reference
        T t1(42);
        cuda_for_dali::std::tuple< T&& > t2(cuda_for_dali::std::move(t1));
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    { // const lvalue reference
        T t1(42);

        cuda_for_dali::std::tuple< T const & > t2(t1);
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);

        cuda_for_dali::std::tuple< T const & > t3(static_cast<T const&>(t1));
        assert(&cuda_for_dali::std::get<0>(t3) == &t1);
    }
    { // lvalue reference
        T t1(42);

        cuda_for_dali::std::tuple< T & > t2(t1);
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    { // const rvalue reference
        T t1(42);

        cuda_for_dali::std::tuple< T const && > t2(cuda_for_dali::std::move(t1));
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    // cuda_for_dali::std::allocator not supported
    /*
    { // rvalue reference via uses-allocator
        T t1(42);
        cuda_for_dali::std::tuple< T&& > t2(cuda_for_dali::std::allocator_arg, A, cuda_for_dali::std::move(t1));
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    { // const lvalue reference via uses-allocator
        T t1(42);

        cuda_for_dali::std::tuple< T const & > t2(cuda_for_dali::std::allocator_arg, A, t1);
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);

        cuda_for_dali::std::tuple< T const & > t3(cuda_for_dali::std::allocator_arg, A, static_cast<T const&>(t1));
        assert(&cuda_for_dali::std::get<0>(t3) == &t1);
    }
    { // lvalue reference via uses-allocator
        T t1(42);

        cuda_for_dali::std::tuple< T & > t2(cuda_for_dali::std::allocator_arg, A, t1);
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    { // const rvalue reference via uses-allocator
        T const t1(42);
        cuda_for_dali::std::tuple< T const && > t2(cuda_for_dali::std::allocator_arg, A, cuda_for_dali::std::move(t1));
        assert(&cuda_for_dali::std::get<0>(t2) == &t1);
    }
    */
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT *can* be constructed from 'tuple<T>'. In this case
    // the 'tuple(UTypes...)' ctor should be chosen and 'UDT' constructed from
    // 'tuple<T>'.
    {
        using VT = ConstructibleFromTupleAndInt;
        cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<VT> t2(t1);
        assert(cuda_for_dali::std::get<0>(t2).state == VT::FromTuple);
    }
    {
        using VT = ConvertibleFromTupleAndInt;
        cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<VT> t2 = {t1};
        assert(cuda_for_dali::std::get<0>(t2).state == VT::FromTuple);
    }
    // Test constructing a 1-tuple of the form tuple<UDT> from another 1-tuple
    // 'tuple<T>' where UDT cannot be constructed from 'tuple<T>' but can
    // be constructed from 'T'. In this case the tuple-like ctor should be
    // chosen and 'UDT' constructed from 'T'
    {
        using VT = ConstructibleFromInt;
        cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<VT> t2(t1);
        assert(cuda_for_dali::std::get<0>(t2).state == VT::FromInt);
    }
    {
        using VT = ConvertibleFromInt;
        cuda_for_dali::std::tuple<int> t1(42);
        cuda_for_dali::std::tuple<VT> t2 = {t1};
        assert(cuda_for_dali::std::get<0>(t2).state == VT::FromInt);
    }

  return 0;
}
