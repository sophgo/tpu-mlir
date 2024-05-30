/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#ifndef VLA_H
#define VLA_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define ALWAYS_INLINE __attribute__((always_inline))
#define INLINE inline 
#define RESTRICT __restrict__
#define VLA_POSTFIX _

#define INDEX1_1(...) ((size_t)SELECT_HEAD(__VA_ARGS__))
#define INDEX1_2(I0, I1, S1) (INDEX1_1(I0) * ((size_t)S1) + (size_t)I1)
#define INDEX1_3(I0, I1, I2, S1, S2) (INDEX1_2(I0, I1, S1) * ((size_t)S2) + (size_t)I2)
#define INDEX1_4(I0, I1, I2, I3, S1, S2, S3) (INDEX1_3(I0, I1, I2, S1, S2) * ((size_t)S3) + (size_t)I3)
#define INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) (INDEX1_4(I0, I1, I2, I3, S1, S2, S3) * ((size_t)S4) + (size_t)I4)
#define INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) (INDEX1_5(I0, I1, I2, I3, I4, S1, S2, S3, S4) * ((size_t)S5) + (size_t)I5)
#define INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) (INDEX1_6(I0, I1, I2, I3, I4, I5, S1, S2, S3, S4, S5) * ((size_t)S6) + (size_t)I6)
#define INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) (INDEX1_7(I0, I1, I2, I3, I4, I5, I6, S1, S2, S3, S4, S5, S6) * ((size_t)S7) + (size_t)I7)
#define INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) (INDEX1_8(I0, I1, I2, I3, I4, I5, I6, I7, S1, S2, S3, S4, S5, S6, S7) * ((size_t)S8) + (size_t)I8)
#define INDEX1_10(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, S1, S2, S3, S4, S5, S6, S7, S8, S9) (INDEX1_9(I0, I1, I2, I3, I4, I5, I6, I7, I8, S1, S2, S3, S4, S5, S6, S7, S8) * ((size_t)S9) + (size_t)I9)

#define EXPAND(...) __VA_ARGS__
#define CONCATENATE2(A, B)  A##B
#define CONCATENATE(A, B) CONCATENATE2(A, B)
#define INDEX1(NDIMS, ...) CONCATENATE(INDEX1_, NDIMS)(__VA_ARGS__)

#define SELECT_HEAD_AUX(A, ...) (A)
#define SELECT_HEAD(...) EXPAND(SELECT_HEAD_AUX(__VA_ARGS__, 0))
#define SELECT_TAIL(A, ...) __VA_ARGS__

#define ACCESS_VLA(NDIMS, ARRAY, ...) CONCATENATE(ARRAY, VLA_POSTFIX)[INDEX1(NDIMS, __VA_ARGS__)]
#define DECLARE_VLA(NDIMS, ELEMENT_TYPE, ARRAY_VAR, ...) \
    ELEMENT_TYPE *RESTRICT CONCATENATE(ARRAY_VAR, VLA_POSTFIX) = SELECT_HEAD(__VA_ARGS__) \
    + 0 * INDEX1(NDIMS, SELECT_TAIL(__VA_ARGS__, SELECT_TAIL(__VA_ARGS__, 0)))

#endif
