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

// template <class ...UTypes>
//    EXPLICIT(...) tuple(UTypes&&...)

// Check that the UTypes... ctor is properly disabled before evaluating any
// SFINAE when the tuple-like copy/move ctor should *clearly* be selected
// instead. This happens 'sizeof...(UTypes) == 1' and the first element of
// 'UTypes...' is an instance of the tuple itself. See PR23256.

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/type_traits>

#include "test_macros.h"


struct UnconstrainedCtor {
  int value_;

  __host__ __device__ UnconstrainedCtor() : value_(0) {}

  // Blows up when instantiated for any type other than int. Because the ctor
  // is constexpr it is instantiated by 'is_constructible' and 'is_convertible'
  // for Clang based compilers. GCC does not instantiate the ctor body
  // but it does instantiate the noexcept specifier and it will blow up there.
  template <typename T>
  __host__ __device__ constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(cuda_for_dali::std::is_same<int, T>::value, "");
  }
};

struct ExplicitUnconstrainedCtor {
  int value_;

  __host__ __device__ ExplicitUnconstrainedCtor() : value_(0) {}

  template <typename T>
  __host__ __device__ constexpr explicit ExplicitUnconstrainedCtor(T value)
    noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value))
  {
      static_assert(cuda_for_dali::std::is_same<int, T>::value, "");
  }

};

int main(int, char**) {
    typedef UnconstrainedCtor A;
    typedef ExplicitUnconstrainedCtor ExplicitA;
    {
        static_assert(cuda_for_dali::std::is_copy_constructible<cuda_for_dali::std::tuple<A>>::value, "");
        static_assert(cuda_for_dali::std::is_move_constructible<cuda_for_dali::std::tuple<A>>::value, "");
        static_assert(cuda_for_dali::std::is_copy_constructible<cuda_for_dali::std::tuple<ExplicitA>>::value, "");
        static_assert(cuda_for_dali::std::is_move_constructible<cuda_for_dali::std::tuple<ExplicitA>>::value, "");
    }
    // cuda_for_dali::std::allocator not supported
    /*
    {
        static_assert(cuda_for_dali::std::is_constructible<
            cuda_for_dali::std::tuple<A>,
            cuda_for_dali::std::allocator_arg_t, cuda_for_dali::std::allocator<void>,
            cuda_for_dali::std::tuple<A> const&
        >::value, "");
        static_assert(cuda_for_dali::std::is_constructible<
            cuda_for_dali::std::tuple<A>,
            cuda_for_dali::std::allocator_arg_t, cuda_for_dali::std::allocator<void>,
            cuda_for_dali::std::tuple<A> &&
        >::value, "");
        static_assert(cuda_for_dali::std::is_constructible<
            cuda_for_dali::std::tuple<ExplicitA>,
            cuda_for_dali::std::allocator_arg_t, cuda_for_dali::std::allocator<void>,
            cuda_for_dali::std::tuple<ExplicitA> const&
        >::value, "");
        static_assert(cuda_for_dali::std::is_constructible<
            cuda_for_dali::std::tuple<ExplicitA>,
            cuda_for_dali::std::allocator_arg_t, cuda_for_dali::std::allocator<void>,
            cuda_for_dali::std::tuple<ExplicitA> &&
        >::value, "");
    }
    */
    {
        cuda_for_dali::std::tuple<A&&> t(cuda_for_dali::std::forward_as_tuple(A{}));
        ((void)t);
        cuda_for_dali::std::tuple<ExplicitA&&> t2(cuda_for_dali::std::forward_as_tuple(ExplicitA{}));
        ((void)t2);
    }

  return 0;
}
