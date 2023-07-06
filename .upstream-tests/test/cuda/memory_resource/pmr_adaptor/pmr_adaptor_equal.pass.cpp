//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <cassert>
#include <cuda_for_dali/memory_resource>
#include <cuda_for_dali/std/cstddef>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/stream_view>
#include <iostream>
#include <memory>
#include <vector>

#if __has_include(<memory_resource>)
#include <memory_resource>
namespace pmr = ::std::pmr;
#elif __has_include(<experimental/memory_resource>)
#include <experimental/memory_resource>
namespace pmr = ::std::experimental::pmr;
#endif

template <typename Kind>
class derived_resource : public cuda_for_dali::memory_resource<Kind> {
public:
private:
  void *do_allocate(cuda_for_dali::std::size_t, cuda_for_dali::std::size_t) override {
    return nullptr;
  }

  void do_deallocate(void *, cuda_for_dali::std::size_t, cuda_for_dali::std::size_t) override {}

  bool
  do_is_equal(cuda_for_dali::memory_resource<cuda_for_dali::memory_kind::host> const &other) const
      noexcept override {
    return dynamic_cast<derived_resource const *>(&other) != nullptr;
  }
};

template <typename Kind>
class more_derived : public derived_resource<Kind> {
public:
private:
  void *do_allocate(cuda_for_dali::std::size_t, cuda_for_dali::std::size_t) override {
    return nullptr;
  }
  void do_deallocate(void *, cuda_for_dali::std::size_t, cuda_for_dali::std::size_t) override {}
};

template <typename T1, typename T2>
void assert_equal(T1 const &lhs, T2 const &rhs) {
  assert(lhs.is_equal(lhs));
  assert(rhs.is_equal(rhs));
  assert(lhs.is_equal(rhs));
  assert(rhs.is_equal(lhs));
}

template <typename P1, typename P2>
void test_equal(cuda_for_dali::pmr_adaptor<P1> const &lhs,
                cuda_for_dali::pmr_adaptor<P2> const &rhs) {
  assert_equal(lhs, rhs);

  pmr::memory_resource const *pmr_lhs{&lhs};
  assert_equal(lhs, *pmr_lhs);
  assert_equal(rhs, *pmr_lhs);

  pmr::memory_resource const *pmr_rhs{&rhs};
  assert_equal(lhs, *pmr_rhs);
  assert_equal(rhs, *pmr_rhs);

  assert_equal(*pmr_rhs, *pmr_lhs);
}

template <typename Kind>
void test_pmr_adaptor_equality(){
  derived_resource<Kind> d;
  cuda_for_dali::pmr_adaptor a_raw{&d};
  cuda_for_dali::pmr_adaptor a_unique{std::make_unique<derived_resource<Kind>>()};
  cuda_for_dali::pmr_adaptor a_shared{std::make_shared<derived_resource<Kind>>()};

  test_equal(a_raw, a_unique);
  test_equal(a_raw, a_shared);
  test_equal(a_unique, a_shared);

  more_derived<Kind> m;
  assert(d.is_equal(m));
  assert(m.is_equal(d));

  cuda_for_dali::pmr_adaptor m_raw{&m};
  test_equal(a_raw, m_raw);
  test_equal(a_unique, m_raw);
  test_equal(a_shared, m_raw);

  cuda_for_dali::pmr_adaptor m_unique{std::make_unique<more_derived<Kind>>()};
  test_equal(a_raw, m_unique);
  test_equal(a_unique, m_unique);
  test_equal(a_shared, m_unique);

  cuda_for_dali::pmr_adaptor m_shared{std::make_shared<more_derived<Kind>>()};
  test_equal(a_raw, m_shared);
  test_equal(a_unique, m_shared);
  test_equal(a_shared, m_shared);
}

int main(int argc, char **argv) {

#ifndef __CUDA_ARCH__
#if defined(_LIBCUDACXX_STD_PMR_NS)
  test_pmr_adaptor_equality<cuda_for_dali::memory_kind::host>();
#endif
#endif

  return 0;
}
