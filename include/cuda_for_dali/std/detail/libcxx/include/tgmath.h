// -*- C++ -*-
//===-------------------------- tgmath.h ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDAFORDALICXX_TGMATH_H
#define _LIBCUDAFORDALICXX_TGMATH_H

/*
    tgmath.h synopsis

#include <ctgmath>

*/

#include <__config>

#if defined(_LIBCUDAFORDALICXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifdef __cplusplus

#include <ctgmath>

#else  // __cplusplus

#include_next <tgmath.h>

#endif  // __cplusplus

#endif  // _LIBCUDAFORDALICXX_TGMATH_H
