// -*- C++ -*-
//===---------------------------- locale.h --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDAFORDALICXX_LOCALE_H
#define _LIBCUDAFORDALICXX_LOCALE_H

/*
    locale.h synopsis

Macros:

    LC_ALL
    LC_COLLATE
    LC_CTYPE
    LC_MONETARY
    LC_NUMERIC
    LC_TIME

Types:

    lconv

Functions:

   setlocale
   localeconv

*/

#include <__config>

#if defined(_LIBCUDAFORDALICXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#include_next <locale.h>

#endif  // _LIBCUDAFORDALICXX_LOCALE_H
