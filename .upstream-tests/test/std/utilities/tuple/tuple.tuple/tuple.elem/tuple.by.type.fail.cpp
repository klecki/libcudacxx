//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <cuda_for_dali/std/tuple>
#include <cuda_for_dali/std/string>
#include "test_macros.h"

struct UserType {};

void test_bad_index() {
    cuda_for_dali::std::tuple<long, long, char, cuda_for_dali::std::string, char, UserType, char> t1;
    TEST_IGNORE_NODISCARD cuda_for_dali::std::get<int>(t1); // expected-error@tuple:* {{type not found}}
    TEST_IGNORE_NODISCARD cuda_for_dali::std::get<long>(t1); // expected-note {{requested here}}
    TEST_IGNORE_NODISCARD cuda_for_dali::std::get<char>(t1); // expected-note {{requested here}}
        // expected-error@tuple:* 2 {{type occurs more than once}}
    cuda_for_dali::std::tuple<> t0;
    TEST_IGNORE_NODISCARD cuda_for_dali::std::get<char*>(t0); // expected-node {{requested here}}
        // expected-error@tuple:* 1 {{type not in empty type list}}
}

void test_bad_return_type() {
    typedef cuda_for_dali::std::unique_ptr<int> upint;
    cuda_for_dali::std::tuple<upint> t;
    upint p = cuda_for_dali::std::get<upint>(t); // expected-error{{deleted copy constructor}}
}

int main(int, char**)
{
    test_bad_index();
    test_bad_return_type();

  return 0;
}
