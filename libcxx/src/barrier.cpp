//===------------------------- barrier.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDAFORDALICXX_HAS_NO_THREADS
#include "barrier"

_LIBCUDAFORDALICXX_BEGIN_NAMESPACE_STD

#if !defined(_LIBCUDAFORDALICXX_HAS_NO_TREE_BARRIER) && !defined(_LIBCUDAFORDALICXX_HAS_NO_THREAD_FAVORITE_BARRIER_INDEX) && (_LIBCUDAFORDALICXX_STD_VER >= 11)

_LIBCUDAFORDALICXX_EXPORTED_FROM_ABI
thread_local ptrdiff_t __libcpp_thread_favorite_barrier_index = 0;

#endif

_LIBCUDAFORDALICXX_END_NAMESPACE_STD

#endif //_LIBCUDAFORDALICXX_HAS_NO_THREADS
