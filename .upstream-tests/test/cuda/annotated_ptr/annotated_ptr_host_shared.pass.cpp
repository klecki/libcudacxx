//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// XFAIL: nvcc

#include "annotated_ptr.h"

int main(int argc, char ** argv)
{
#ifndef __CUDA_ARCH__
    cuda_for_dali::access_property ap(cuda_for_dali::access_property::persisting{});
    int* array0 = new int[9];
    cuda_for_dali::annotated_ptr<int, cuda_for_dali::access_property> array_anno_ptr{array0, ap};
    cuda_for_dali::annotated_ptr<int, cuda_for_dali::access_property::shared> shared_ptr;

    array_anno_ptr = shared_ptr;  //  fail to compile, as expected
#endif

    return 0;
}
