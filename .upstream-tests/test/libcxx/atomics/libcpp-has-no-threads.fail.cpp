//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/atomic>

// Test that including <cuda/std/atomic> fails to compile when _LIBCUDAFORDALICXX_HAS_NO_THREADS
// is defined.

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

// nvcc doesn't propagate pgi's or icc's preprocessor failures
// UNSUPPORTED: pgi, icc

// MODULES_DEFINES: _LIBCUDAFORDALICXX_HAS_NO_THREADS
#ifndef _LIBCUDAFORDALICXX_HAS_NO_THREADS
#define _LIBCUDAFORDALICXX_HAS_NO_THREADS
#endif

#include <cuda_for_dali/std/atomic>

int main(int, char**)
{

  return 0;
}
