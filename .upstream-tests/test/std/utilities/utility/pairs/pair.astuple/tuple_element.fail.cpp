//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <cuda_for_dali/std/utility>

int main(int, char**)
{
    typedef cuda_for_dali::std::pair<int, short> T;
    cuda_for_dali::std::tuple_element<2, T>::type foo; // expected-error@utility:* {{Index out of bounds in cuda_for_dali::std::tuple_element<cuda_for_dali::std::pair<T1, T2>>}}

  return 0;
}
