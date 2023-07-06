//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> pair<V1, V2> make_pair(T1&&, T2&&);

#include <cuda_for_dali/std/utility>
// cuda/std/memory not supported
// #include <cuda_for_dali/std/memory>
#include <cuda_for_dali/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef cuda_for_dali::std::pair<int, short> P1;
        P1 p1 = cuda_for_dali::std::make_pair(3, static_cast<short>(4));
        assert(p1.first == 3);
        assert(p1.second == 4);
    }

#if TEST_STD_VER >= 11
    // cuda/std/memory not supported
    /*
    {
        typedef cuda_for_dali::std::pair<cuda_for_dali::std::unique_ptr<int>, short> P1;
        P1 p1 = cuda_for_dali::std::make_pair(cuda_for_dali::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
        assert(*p1.first == 3);
        assert(p1.second == 4);
    }
    {
        typedef cuda_for_dali::std::pair<cuda_for_dali::std::unique_ptr<int>, short> P1;
        P1 p1 = cuda_for_dali::std::make_pair(nullptr, static_cast<short>(4));
        assert(p1.first == nullptr);
        assert(p1.second == 4);
    }
    */
#endif
#if TEST_STD_VER >= 14
    {
        typedef cuda_for_dali::std::pair<int, short> P1;
        constexpr P1 p1 = cuda_for_dali::std::make_pair(3, static_cast<short>(4));
        static_assert(p1.first == 3, "");
        static_assert(p1.second == 4, "");
    }
#endif


  return 0;
}
