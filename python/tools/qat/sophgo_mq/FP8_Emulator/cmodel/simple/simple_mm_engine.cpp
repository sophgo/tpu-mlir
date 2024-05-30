/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <immintrin.h>

/* column major dat format */
void MMEngine_avx2_ps(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc)
{
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i+=8) {
      __m256 cij = _mm256_loadu_ps(&C[ j*ldc + i]);
      for (int l = 0; l < k; l++) {
        __m256 aik = _mm256_loadu_ps(&A[ i + l * lda]);
        __m256 bkj = _mm256_broadcast_ss(&B[ l + j * ldb]);
        cij = _mm256_add_ps(cij, _mm256_mul_ps(aik, bkj));
      }
      _mm256_storeu_ps(&C[ j * ldc + i ], cij);
    }
  }
}

/* column major dat format */
void MMEngine_strideB_avx2_ps(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc, int strideB)
{
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i+=8) {
      __m256 cij = _mm256_loadu_ps(&C[ j*ldc + i]);
      for (int l = 0; l < k; l++) {
        __m256 aik = _mm256_loadu_ps(&A[ i + l * lda]);
        __m256 bkj = _mm256_broadcast_ss(&B[ l*strideB + j * ldb]);
        cij = _mm256_add_ps(cij, _mm256_mul_ps(aik, bkj));
      }
      _mm256_storeu_ps(&C[ j * ldc + i], cij);
    }
  }
}

