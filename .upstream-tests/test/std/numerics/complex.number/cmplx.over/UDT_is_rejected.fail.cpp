//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

// <cuda/std/complex>

// Test that UDT's convertible to an integral or floating point type do not
// participate in overload resolution.

#include <cuda_for_dali/std/complex>
#include <cuda_for_dali/std/type_traits>
#include <cuda_for_dali/std/cassert>

template <class IntT>
struct UDT {
  operator IntT() const { return 1; }
};

UDT<float> ft;
UDT<double> dt;
// CUDA treats long double as double
// UDT<long double> ldt;
UDT<int> it;
UDT<unsigned long> uit;

int main(int, char**)
{
    {
        cuda_for_dali::std::real(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::real(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::real(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::real(it); // expected-error {{no matching function}}
        cuda_for_dali::std::real(uit); // expected-error {{no matching function}}
    }
    {
        cuda_for_dali::std::imag(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::imag(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::imag(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::imag(it); // expected-error {{no matching function}}
        cuda_for_dali::std::imag(uit); // expected-error {{no matching function}}
    }
    {
        cuda_for_dali::std::arg(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::arg(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::arg(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::arg(it); // expected-error {{no matching function}}
        cuda_for_dali::std::arg(uit); // expected-error {{no matching function}}
    }
    {
        cuda_for_dali::std::norm(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::norm(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::norm(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::norm(it); // expected-error {{no matching function}}
        cuda_for_dali::std::norm(uit); // expected-error {{no matching function}}
    }
    {
        cuda_for_dali::std::conj(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::conj(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::conj(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::conj(it); // expected-error {{no matching function}}
        cuda_for_dali::std::conj(uit); // expected-error {{no matching function}}
    }
    {
        cuda_for_dali::std::proj(ft); // expected-error {{no matching function}}
        cuda_for_dali::std::proj(dt); // expected-error {{no matching function}}
        // cuda_for_dali::std::proj(ldt); // expected-error {{no matching function}}
        cuda_for_dali::std::proj(it); // expected-error {{no matching function}}
        cuda_for_dali::std::proj(uit); // expected-error {{no matching function}}
    }

  return 0;
}
