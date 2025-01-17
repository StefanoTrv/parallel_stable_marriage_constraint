/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

#ifndef __FAIL_H
#define __FAIL_H

#ifdef  __MINGW64__
    #define SETJUMP(x) setjmp(x)
    #define LONGJMP(x,y) longjmp(x,y)
#else
    #define SETJUMP(x) _setjmp(x)
    #define LONGJMP(x,y) _longjmp(x,y)
#endif

enum Status { Failure,Success,Suspend };

//#define CPPFAIL 1
#if defined(CPPFAIL)
static inline void failNow() {
    throw Failure;
}

#define TRYFAIL try {
#define ONFAIL  } catch(...) {
#define ENDFAIL }

#else

#include <stdio.h>
#include <setjmp.h>

extern __thread jmp_buf* ptr;

#define TRYFAIL  { \
   jmp_buf buf; \
   jmp_buf* old = ptr; \
   int st = SETJUMP(buf); \
   if (st==0) { \
      ptr = &buf;

#define ONFAIL ptr = old; \
     } else {             \
      ptr = old;

#define ENDFAIL }}

static inline void failNow() {
    LONGJMP(*ptr,1);
}

#endif

#endif
