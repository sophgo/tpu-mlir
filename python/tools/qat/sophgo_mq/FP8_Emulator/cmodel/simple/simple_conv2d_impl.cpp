/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)

#include <omp.h>
#endif
#include <vla.h>

#define CHANNEL_BLOCK 64

extern void MMEngine_avx2_ps(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc);
extern void MMEngine_strideB_avx2_ps(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc, int strideB);

typedef struct {
  int nImg;
  int ifm;
  int nBIfm;
  int nbIfm;
  int nBOfm;
  int nbOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  /* additional padding arams for feature maps -- used in WU kernel*/
  int pad_iw;
  int pad_ow;
  int nbofw;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} gemm_conv_t;

INLINE void zero_buf(float* buf, long size) {
  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

INLINE void copy_buf(float* src, float* dst, long size) {
  int i;
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
#if defined(_OPENMP)
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
  }
}

INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  DECLARE_VLA(4, float, input, nchw, C, H, W);
  int n, h, w, c;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if(h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            ACCESS_VLA(4,  input, n, c, h, w, C, H, W) = 0.0;
        }
      }
    }
  }
}

INLINE 
void copy_NCHW_to_GEMM(const float* nchw, float* gemm, const int N, const int H, const int W, const int C, const int cblock)
{
  DECLARE_VLA(5,       float, output, gemm, C/cblock, H, W, cblock);
  DECLARE_VLA(4, const float,  input, nchw, C, H, W);
  int n, h, w, c1, c2;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c1, c2)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/cblock; c1++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( c2 = 0; c2 < cblock; c2++ ) {
            ACCESS_VLA(5, output, n, c1, h, w, c2, C/cblock, H, W, cblock) =
              ACCESS_VLA(4,  input, n, (c1*cblock)+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}

INLINE 
void copy_pad_NCHW_to_GEMM(const float* nchw, float* gemm, const int N, const int H, const int W, const int C, const int cblock, 
		int pad_h, int pad_w, int pad_c)
{
  int HP, WP;
  HP = H + 2*pad_h; 
  WP = W + 2*pad_w; 
  int CP = C + pad_c;
  DECLARE_VLA(5,       float, output, gemm, CP/cblock, HP, WP, cblock);
  DECLARE_VLA(4, const float,  input, nchw, C, H, W);
  int n, h, w, c1, c2;
  /* if channels are smaller than cblock, use channels as the cblock */
  int lcblock = cblock;
  if (C < cblock) lcblock = C;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c1, c2)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/lcblock; c1++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( c2 = 0; c2 < lcblock; c2++ ) {
            ACCESS_VLA(5, output, n, c1, h+pad_h, w+pad_w, c2, C/cblock, HP, WP, cblock) =
              ACCESS_VLA(4,  input, n, (c1*lcblock)+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}

INLINE 
void copy_pad_NCHW_to_GEMM_ex(const float* nchw, float* gemm, const int N, const int H, const int W, const int C, const int cblock, 
		int pad_h, int pad_w, int pad_wex, int pad_c)
{
  int HP, WP;
  HP = H + 2*pad_h; 
  WP = W + 2*pad_w + pad_wex; 
  int CP = C + pad_c;
  DECLARE_VLA(5,       float, output, gemm, CP/cblock, HP, WP, cblock);
  DECLARE_VLA(4, const float,  input, nchw, C, H, W);
  int n, h, w, c1, c2;
  /* if channels are smaller than cblock, use channels as the cblock */
  int lcblock = cblock;
  if (C < cblock) lcblock = C;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c1, c2)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/lcblock; c1++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( c2 = 0; c2 < lcblock; c2++ ) {
            ACCESS_VLA(5, output, n, c1, h+pad_h, w+pad_w, c2, C/cblock, HP, WP, cblock) =
              ACCESS_VLA(4,  input, n, (c1*lcblock)+c2, h, w, C, H, W);
          }
        }
      }
    }
  }
}


INLINE 
void copy_GEMM_to_NCHW(const float* gemm, float* nchw, const int N, const int H, const int W, const int C, const int cblock)
{
  DECLARE_VLA(5, const float,  input, gemm, C/cblock, H, W, cblock);
  DECLARE_VLA(4,       float, output, nchw, C, H, W);
  int n, h, w, c1, c2;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c1, c2)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/cblock; c1++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( c2 = 0; c2 < cblock; c2++ ) {
            ACCESS_VLA(4,  output, n, (c1*cblock)+c2, h, w, C, H, W) =
              ACCESS_VLA(5, input, n, c1, h, w, c2, C/cblock, H, W, cblock);
          }
        }
      }
    }
  }
}
INLINE 
void copy_pad_GEMM_to_NCHW(const float* gemm, float* nchw, const int N, const int H, const int W, const int C, const int cblock, 
		int pad_h, int pad_w, int pad_c)
{
  int HP, WP;
  HP = H + 2*pad_h;
  WP = W + 2*pad_w;
  int CP = C + pad_c;

  DECLARE_VLA(5, const float,  input, gemm, CP/cblock, HP, WP, cblock);
  DECLARE_VLA(4,       float, output, nchw, C, H, W);
  int n, h, w, c1, c2;

  /* if number of channles are smaller than cblock, use C as cblock */
  int lcblock = cblock;
  if (cblock > C) lcblock = C;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(n, h, w, c1, c2)
#endif
  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/lcblock; c1++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( c2 = 0; c2 < lcblock; c2++ ) {
            ACCESS_VLA(4,  output, n, (c1*lcblock)+c2, h, w, C, H, W) =
              ACCESS_VLA(5, input, n, c1, h+pad_h, w+pad_w, c2, C/cblock, HP, WP, cblock);
          }
        }
      }
    }
  }
}

INLINE 
void copy_KCRS_to_GEMM(const float* kcrs, float* gemm, const int R, const int S, const int C, const int K, const int cblock, const int kblock)
{
  DECLARE_VLA(6,       float, output, gemm, C/cblock, R, S, cblock, kblock);
  DECLARE_VLA(4, const float,  input, kcrs, C, R, S);
  int r, s, c1, c2, k1, k2;
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(r, s, c1, c2, k1, k2)
#endif
  for ( k1 = 0; k1 < K/kblock; k1++ ) {
    for ( c1 = 0; c1 < C/cblock; c1++ ) {
      for ( r = 0; r < R; r++ ) {
        for ( s = 0; s < S; s++ ) {
          for ( c2 = 0; c2 < cblock; c2++ ) {
            for ( k2 = 0; k2 < kblock; k2++ ) {
              ACCESS_VLA(6, output, k1, c1, r, s, c2, k2, C/cblock, R, S, cblock, kblock) =
                ACCESS_VLA(4,  input, (k1*kblock)+k2, (c1*cblock)+c2, r, s, C, R, S);
            }
          }
        }
      }
    }
  }
}

INLINE 
void copy_pad_KCRS_to_GEMM(const float* kcrs, float* gemm, const int R, const int S, const int C, const int K, const int cblock, 
		const int kblock, int pad_c, int pad_k)
{
  int CP = C + pad_c;
  int KP = K + pad_k;
  int lcblock = cblock;
  int lkblock = kblock;
  if (C < cblock) lcblock = C; 
  if (K < kblock) lkblock = K; 

  DECLARE_VLA(6,       float, output, gemm, CP/cblock, R, S, cblock, kblock);
  DECLARE_VLA(4, const float,  input, kcrs, C, R, S);
  int r, s, c1, c2, k1, k2;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(r, s, c1, c2, k1, k2) shared(CP, KP, lcblock, lkblock)
#endif
  for ( k1 = 0; k1 < KP/kblock; k1++ ) {
    for ( c1 = 0; c1 < CP/cblock; c1++ ) {
      for ( r = 0; r < R; r++ ) {
        for ( s = 0; s < S; s++ ) {
          for ( c2 = 0; c2 < lcblock; c2++ ) {
            for ( k2 = 0; k2 < lkblock; k2++ ) {
              ACCESS_VLA(6, output, k1, c1, r, s, c2, k2, CP/cblock, R, S, cblock, kblock) =
                ACCESS_VLA(4,  input, (k1*lkblock)+k2, (c1*lcblock)+c2, r, s, C, R, S);
            }
          }
        }
      }
    }
  }
}


INLINE 
void copy_GEMM_to_KCRS(const float* gemm, float* kcrs, const int R, const int S, const int C, const int K, const int cblock, const int kblock)
{
  DECLARE_VLA(6, const float,  input, gemm, C/cblock, R, S, cblock, kblock);
  DECLARE_VLA(4,       float, output, kcrs, C, R, S);
  int r, s, c1, c2, k1, k2;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(r, s, c1, c2, k1, k2)
#endif
  for ( k1 = 0; k1 < K/kblock; k1++ ) {
    for ( c1 = 0; c1 < C/cblock; c1++ ) {
      for ( r = 0; r < R; r++ ) {
        for ( s = 0; s < S; s++ ) {
          for ( c2 = 0; c2 < cblock; c2++ ) {
            for ( k2 = 0; k2 < kblock; k2++ ) {
              ACCESS_VLA(4,  output, (k1*kblock)+k2, (c1*cblock)+c2, r, s, C, R, S) = 
                ACCESS_VLA(6, input, k1, c1, r, s, c2, k2, C/cblock, R, S, cblock, kblock);
            }
          }
        }
      }
    }
  }
}

INLINE 
void copy_pad_GEMM_to_KCRS(const float* gemm, float* kcrs, const int R, const int S, const int C, const int K, const int cblock, 
		const int kblock, int pad_c, int pad_k)
{

  int CP = C + pad_c;
  int KP = K + pad_k;
  int lcblock = cblock;
  int lkblock = kblock;
  if (C < cblock) lcblock = C; 
  if (K < kblock) lkblock = K; 

  DECLARE_VLA(6, const float, input, gemm, CP/cblock, R, S, cblock, kblock);
  DECLARE_VLA(4,       float,  output, kcrs, C, R, S);
  int r, s, c1, c2, k1, k2;

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(r, s, c1, c2, k1, k2) shared(CP, KP, lcblock, lkblock)
#endif
  for ( k1 = 0; k1 < KP/kblock; k1++ ) {
    for ( c1 = 0; c1 < CP/cblock; c1++ ) {
      for ( r = 0; r < R; r++ ) {
        for ( s = 0; s < S; s++ ) {
          for ( c2 = 0; c2 < lcblock; c2++ ) {
            for ( k2 = 0; k2 < lkblock; k2++ ) {
              ACCESS_VLA(4,  output, (k1*lkblock)+k2, (c1*lcblock)+c2, r, s, C, R, S) =
                ACCESS_VLA(6, input, k1, c1, r, s, c2, k2, CP/cblock, R, S, cblock, kblock);
            }
          }
        }
      }
    }
  }
}

INLINE 
void gemm_kernel_conv_fp(gemm_conv_t* param, float* output, float* input, float* filter, const float* bias)
{
  int nImg     = param->nImg;
  int nBIfm     = param->nBIfm;
  int nbIfm     = param->nbIfm;
  int nBOfm     = param->nBOfm;
  int nbOfm     = param->nbOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  //int ifh       = param->ifh;
  //int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  //int nbofw     = param->nbofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm1, ifm1, oj, oi, ij, kj, ki, ofm2;
#ifdef NAIVE_CODE
  int ii, ifm2;
#endif
  DECLARE_VLA(5, float, output_t, output + (pad_h * ofwp * nbOfm + pad_w * nbOfm), nBOfm, ofhp, ofwp, nbOfm);
  DECLARE_VLA(5, float,  input_t,  input                                         , nBIfm, ifhp, ifwp, nbIfm);
  DECLARE_VLA(6, float, filter_t, filter, nBIfm, kh, kw, nbIfm, nbOfm);

  int lda = CHANNEL_BLOCK;
  int ldb = stride_w*CHANNEL_BLOCK;
  int ldc = CHANNEL_BLOCK;
#if 0 
  /* compute new n-blocking to fit limitation of N<=16 of TMUL */
  for ( oii = 16; oii > 0; --oii ) {
    if ( ofw % oii == 0 ) {
      nbofw = oii;
      break;
    }
  }
#endif
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) private(img, ofm1, ofm2, oj, oi )
#endif
  /* pre-initializing with bias */ 
  for (img = 0; img < nImg; ++img) {
    for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          for (ofm2 = 0; ofm2 < nbOfm; ++ofm2) {
            ACCESS_VLA(5, output_t, img, ofm1, oj, oi, ofm2, nBOfm, ofhp, ofwp, nbOfm) = bias[ofm1*nbOfm + ofm2];
          }
        }
      }
    }
  }

#if defined(_OPENMP)
#ifdef NAIVE_CODE
#pragma omp parallel for collapse(4) private(img, ofm1, ofm2, ifm1, ifm2, oj, oi, ij, ii, kj, ki)
#else
#pragma omp parallel for collapse(4) private(img, ofm1, ofm2, ifm1, oj, oi, ij, kj, ki)
#endif
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
      for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (kj = 0; kj < kh; ++kj) {
            for (ki = 0; ki < kw; ++ki) {
              /* let's do a 64 x ofw x 64 GEMM : M=nbOfm, N=ofw, K=nbIfm (col-major) */
#ifdef NAIVE_CODE
              for (oi = 0; oi < ofw; ++oi) {
                ii = oi * stride_w;
                for (ofm2 = 0; ofm2 < nbOfm; ++ofm2) {
                  for (ifm2 = 0; ifm2 < nbIfm; ++ifm2) {
                    ACCESS_VLA(5, output_t,  img, ofm1,      oj, oi, ofm2, nBOfm, ofhp, ofwp, nbOfm)           += /* C */
                        ACCESS_VLA(6, filter_t, ofm1, ifm1,  kj, ki, ifm2,  ofm2, nBIfm,   kh, kw, nbIfm, nbOfm)  /* A */
                      * ACCESS_VLA(5,  input_t,  img, ifm1, ij + kj, ii + ki, ifm2, nBIfm,  ifhp, ifwp, nbIfm);   /* B */
                  }
                }
              }
#else
              MMEngine_avx2_ps (nbOfm, ofw, nbIfm, 1.0,
                     &ACCESS_VLA(6, filter_t, ofm1, ifm1,      kj, ki, 0, 0, nBIfm, kh, kw, nbIfm, nbOfm), lda,
                     &ACCESS_VLA(5,  input_t,  img, ifm1, ij + kj, ki, 0, nBIfm, ifhp, ifwp, nbIfm),       ldb, 1.0,
                     &ACCESS_VLA(5, output_t,  img, ofm1,      oj, 0, 0, nBOfm, ofhp, ofwp, nbOfm),       ldc);
#endif
            }
          }
        }
      }
    }
  }
}

INLINE void gemm_kernel_conv_bp(gemm_conv_t* param, float* input, float* output, float* filter, float* tr_filter)
{
  int nImg      = param->nImg;
  int nBIfm     = param->nBIfm;
  int nbIfm     = param->nbIfm;
  int nBOfm     = param->nBOfm;
  int nbOfm     = param->nbOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  //int ifh       = param->ifh;
  //int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  //int nbofw     = param->nbofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
 
  /* loop counters */
  int img, ofm1, ifm1, ofm2, ifm2, oj, ij, kj, ki;/*, oii;*/
#ifdef NAIVE_CODE
  int ii, oi;
#endif
  DECLARE_VLA(5, float, output_t, output + (pad_h * ofwp * nbOfm + pad_w * nbOfm), nBOfm, ofhp, ofwp, nbOfm);
  DECLARE_VLA(5, float,  input_t,  input ,                                         nBIfm, ifhp, ifwp, nbIfm);
  DECLARE_VLA(6, float, filter_t, filter, nBIfm, kh, kw, nbIfm, nbOfm);
  DECLARE_VLA(6, float, tr_filter_t, tr_filter, nBIfm, kh, kw, nbOfm, nbIfm);

#if defined(_OPENMP)
# pragma omp parallel for collapse(4) private(ofm1, ifm1, kj, ki, ofm2, ifm2)
#endif
  for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
    for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
      for (kj = 0; kj < kh; ++kj) {
        for (ki = 0; ki < kw; ++ki) {
          for (ofm2 = 0; ofm2 < nbOfm; ++ofm2) {
            for (ifm2 = 0; ifm2 < nbIfm; ++ifm2) {
              ACCESS_VLA(6, tr_filter_t, ofm1, ifm1, kj, ki, ofm2, ifm2, nBIfm, kh,   kw,   nbOfm, nbIfm) =
                ACCESS_VLA( 6, filter_t, ofm1, ifm1, kj, ki, ifm2, ofm2, nBIfm, kh,   kw,   nbIfm, nbOfm);
            }
          }
        }
      }
    }
  }

  int lda = CHANNEL_BLOCK;
  int ldb = CHANNEL_BLOCK;
  int ldc = stride_w*CHANNEL_BLOCK;
#if 0
  /* compute new n-blocking  */
  for ( oii = 16; oii > 0; --oii ) {
    if ( ofw % oii == 0 ) {
      nbofw = oii;
      break;
    }
  }
#endif
#if defined(_OPENMP)
#ifdef NAIVE_CODE
# pragma omp parallel for collapse(4) private(img, ofm1, ifm1, ofm2, ifm2, oj, oi, ij, ii, kj, ki)
#else
# pragma omp parallel for collapse(4) private(img, ofm1, ifm1, ofm2, ifm2, oj, ij, kj, ki)
#endif
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
      for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (kj = 0; kj < kh; ++kj) {
            for (ki = 0; ki < kw; ++ki) {
#ifdef NAIVE_CODE
              for (ifm2 = 0; ifm2 < nbIfm; ++ifm2) {
                for (oi = 0; oi < ofw; ++oi) {
                  ii = oi * stride_w;
                  for (ofm2 = 0; ofm2 < nbOfm; ++ofm2) {
                    ACCESS_VLA(5,     input_t,  img, ifm1, ij + kj, ii + ki, ifm2, nBIfm, ifhp, ifwp, nbIfm)        += /* C */
                        ACCESS_VLA(6, tr_filter_t, ofm1, ifm1,      kj, ki, ofm2, ifm2, nBIfm, kh, kw, nbOfm, nbIfm)   /* A */
                      * ACCESS_VLA(5,    output_t,  img, ofm1,      oj, oi, ofm2, nBOfm, ofhp, ofwp, nbOfm);           /* B */
                  }
                }
              }
#else
              MMEngine_avx2_ps ( nbIfm, ofw, nbOfm, 1.0,
                      &ACCESS_VLA(6, tr_filter_t, ofm1, ifm1,      kj, ki, 0, 0, nBIfm, kh, kw, nbOfm, nbIfm), lda,
                      &ACCESS_VLA(5,    output_t,  img, ofm1,      oj, 0, 0, nBOfm, ofhp, ofwp, nbOfm),       ldb, 1.0,
                      &ACCESS_VLA(5,     input_t,  img, ifm1, ij + kj, ki, 0, nBIfm, ifhp, ifwp, nbIfm),       ldc);
#endif
            }
          }
        }
      }
    }
  }
}

INLINE void gemm_kernel_conv_wu(gemm_conv_t* param, float* filter, float* output, float* input, float* tr_input)
{
  int nImg      = param->nImg;
  int nBIfm     = param->nBIfm;
  int nbIfm     = param->nbIfm;
  int nBOfm     = param->nBOfm;
  int nbOfm     = param->nbOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  //int nbofw     = param->nbofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  //int pad_iw    = param->pad_iw;
  int pad_ow    = param->pad_ow;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;

  /* loop counters */
  int img, ofm1, ifm1, ifm2, oj, ij, ii, kj, ki;/*, oii;*/
#ifdef NAIVE_CODE
  int ofm2, oi;
#endif

  DECLARE_VLA(5, float, output_t, output + (pad_h * ofwp * nbOfm + pad_w * nbOfm), nBOfm, ofhp, ofwp, nbOfm);
  DECLARE_VLA(5, float,  input_t,  (float*)input,                                          nBIfm, ifhp, ifwp, nbIfm);
  DECLARE_VLA(5, float, tr_input_t, tr_input,                                      nBIfm, ifhp+(2*pad_h), nbIfm, ifwp+(2*pad_w));
  DECLARE_VLA(6, float, filter_t, filter, nBIfm, kh, kw, nbIfm, nbOfm);

#if defined(_OPENMP)
# pragma omp parallel for collapse(2) private(img, ifm1, ij, ii, ifm2)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
      for (ij = 0; ij < ifh; ++ij) {
        for (ifm2 = 0; ifm2 < nbIfm; ++ifm2) {
          for (ii = 0; ii < ifw; ++ii) {
            ACCESS_VLA(5,  tr_input_t, img, ifm1, ij+pad_h, ifm2,   ii+pad_w, nBIfm, ifhp+(2*pad_h), nbIfm, ifwp+(2*pad_w)) = 
              ACCESS_VLA(5,   input_t, img, ifm1, ij+pad_h,   ii+pad_w, ifm2, nBIfm, ifhp, ifwp, nbIfm);
          }
        }
      }
    }
  }

  int lda = CHANNEL_BLOCK;
  int ldb = ifwp + (2*pad_w); 
  int ldc = CHANNEL_BLOCK;
#if 0
  int nbifm = nbIfm;
  for ( oii = 16; oii > 0; --oii ) {
    if ( nbIfm % oii == 0 ) {
      nbifm = oii;
      break;
    }
  }
#endif
#if defined(_OPENMP)
#ifdef NAIVE_CODE
# pragma omp parallel for collapse(2) private(img, ofm1, ifm1, ofm2, ifm2, oj, oi, ij, ii, kj, ki)
#else
# pragma omp parallel for collapse(2) private(img, ofm1, ifm1, ifm2, oj, ij, ii, kj, ki)
#endif
#endif
  for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
    for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
      for (img = 0; img < nImg; ++img) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (kj = 0; kj < kh; ++kj) {
            for (ki = 0; ki < kw; ++ki) {
#ifdef NAIVE_CODE
              for (ifm2 = 0; ifm2 < nbIfm; ++ifm2) {
                for (oi = 0; oi < ofw; ++oi) {
                  ii = oi * stride_w;
                  for (ofm2 = 0; ofm2 < nbOfm; ++ofm2) {
                    ACCESS_VLA(    6,   filter_t, ofm1, ifm1,      kj,   ki,    ifm2, ofm2, nBIfm, kh,   kw,   nbIfm, nbOfm)              /* C */ +=
                        ACCESS_VLA(5,   output_t, img,  ofm1,      oj,   oi,    ofm2,       nBOfm, ofhp, ofwp, nbOfm)                     /* A */
                      * ACCESS_VLA(5, tr_input_t, img,  ifm1, ij + kj, ifm2, ii + ki,       nBIfm, ifhp+(2*pad_h), nbIfm, ifwp+(2*pad_w)) /* B */;
                  }
                }
              }
#else
              MMEngine_strideB_avx2_ps (nbOfm, nbIfm, ofw+pad_ow, 1.0,
                        &ACCESS_VLA(5,   output_t,  img, ofm1,      oj,  0,  0, nBOfm,            ofhp,  ofwp,          nbOfm), lda,
                        &ACCESS_VLA(5, tr_input_t,  img, ifm1, ij + kj,  0, ki, nBIfm,  ifhp+(2*pad_h), nbIfm, ifwp+(2*pad_w)), ldb, 1.0,
                        &ACCESS_VLA(6,   filter_t, ofm1, ifm1,      kj, ki,  0,  0, nBIfm, kh, kw, nbIfm, nbOfm), ldc, stride_w);
#endif
            }
          }
        }
      }
    }
  }
}

int simple_conv2d_impl_fp(float* outputs, float *inputs, float *weights, float* bias, int N, int C, int iH, int iW, 
	int K, int R, int S, int stride, int padding, int dilation, int groups) 
{
  float *gemm_input, *gemm_output, *gemm_filter;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w;

  int ifw = iW;         /* input width, "W" */
  int ifh = iH;         /* input height, "H" */
  int nImg = N;         /* mini-batch size, "N" */
  int nIfm = C;         /* number of input feature maps, "C" */
  int nOfm = K;         /* number of output feature maps, "K" */
  int kh = R;           /* filter height, "R" */
  int kw = S;           /* filter width, "S" */
  pad_h = padding;      /* padding in output */
  pad_w = padding;      /* padding in output */

  /* apply stride in both dimensions */
  stride_w = stride;
  stride_h = stride;

  /* deriving some values image size */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  //ofh = (ifh + 2 * pad_h - dilation * (kh-1) - 1) / stride_h + 1;
  //ofw = (ifw + 2 * pad_w - dilation * (kw-1) - 1) / stride_w + 1;

  ifhp = ifh + 2 * pad_h;
  ifwp = ifw + 2 * pad_w;
  ofhp = ofh + 2 * pad_h;
  ofwp = ofw + 2 * pad_w;

  /*pad nIfm and nOfm to multiples of 64 */ 
  int ifm_pad = 0;
  int ofm_pad = 0;
  int nIfmp = nIfm;
  int nOfmp = nOfm;
  if (nIfm % 64 != 0){ 
      ifm_pad = (64-(nIfm%64));
      nIfmp += ifm_pad; 
  }
  if (nOfm % 64 != 0) {
      ofm_pad = (64-(nOfm%64));
      nOfmp += ofm_pad;  
  }

  gemm_conv_t gemm_param;
  /* set struct for naive convolution */
  gemm_param.nImg = nImg;
  gemm_param.nBIfm = nIfmp/CHANNEL_BLOCK;
  gemm_param.nbIfm = CHANNEL_BLOCK;
  gemm_param.nBOfm = nOfmp/CHANNEL_BLOCK;
  gemm_param.nbOfm = CHANNEL_BLOCK;
  gemm_param.ifhp = ifhp;
  gemm_param.ifwp = ifwp;
  gemm_param.ofhp = ofhp;
  gemm_param.ofwp = ofwp;
  gemm_param.ifh = ifh;
  gemm_param.ifw = ifw;
  gemm_param.ofh = ofh;
  gemm_param.ofw = ofw;
  gemm_param.pad_h = pad_h;
  gemm_param.pad_w = pad_w;

  if ( ofw == 56 ) {
    gemm_param.nbofw = 28;
  } else {
    gemm_param.nbofw = ofw;
  }
  gemm_param.kh = kh;
  gemm_param.kw = kw;
  gemm_param.stride_h = stride_h;
  gemm_param.stride_w = stride_w;

  gemm_input            = (float*)_mm_malloc( nImg*nIfmp*ifhp*ifwp*sizeof(float), 2097152);
  gemm_filter           = (float*)_mm_malloc( nOfmp*nIfmp*kh*kw*    sizeof(float), 2097152);
  gemm_output           = (float*)_mm_malloc( nImg*nOfmp*ofhp*ofwp*sizeof(float), 2097152);
  zero_buf(gemm_input, nImg*nIfmp*ifhp*ifwp);
  if (nIfm % 64 != 0 || nOfm % 64 != 0){ 
     zero_buf(gemm_filter, nOfmp*nIfmp*kh*kw);
     zero_buf(gemm_output, nImg*nOfmp*ofhp*ofwp);
  }

  /* copy data into GEMM optimized format */
  copy_pad_NCHW_to_GEMM(inputs, gemm_input, nImg, ifh, ifw, nIfm, CHANNEL_BLOCK, pad_h, pad_w, ifm_pad);
  copy_pad_KCRS_to_GEMM(weights, gemm_filter, kh, kw, nIfm, nOfm, CHANNEL_BLOCK, CHANNEL_BLOCK, ifm_pad, ofm_pad);
  gemm_kernel_conv_fp(&gemm_param, gemm_output, gemm_input, gemm_filter, bias);
  /* copy out data */
  copy_pad_GEMM_to_NCHW(gemm_output, outputs, nImg, ofh, ofw, nOfm, CHANNEL_BLOCK, pad_h, pad_w, ofm_pad);

  _mm_free(gemm_input);
  _mm_free(gemm_filter);
  _mm_free(gemm_output);
  return 0;
}

int simple_conv2d_impl_bp(float* inputs, float *outputs, float *weights, int N, int C, int iH, int iW, 
	int K, int R, int S, int stride, int padding, int dilation, int groups) 
{
  float *gemm_input, *gemm_output, *gemm_filter, *gemm_filter_tr;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w;

  int ifw = iW;         /* input width, "W" */
  int ifh = iH;         /* input height, "H" */
  int nImg = N;         /* mini-batch size, "N" */
  int nIfm = C;         /* number of input feature maps, "C" */
  int nOfm = K;         /* number of output feature maps, "K" */
  int kh = R;           /* filter height, "R" */
  int kw = S;           /* filter width, "S" */
  pad_h = padding;      /* padding in output */
  pad_w = padding;      /* padding in output */

  /* apply stride in both dimensions */
  stride_w = stride;
  stride_h = stride;
  /* deriving some values image size */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  //ofh = (ifh + 2 * pad_h - dilation * (kh-1) - 1) / stride_h + 1;
  //ofw = (ifw + 2 * pad_w - dilation * (kw-1) - 1) / stride_w + 1;
  ifhp = ifh + 2 * pad_h;
  ifwp = ifw + 2 * pad_w;
  ofhp = ofh + 2 * pad_h;
  ofwp = ofw + 2 * pad_w;
  /*pad nIfm and nOfm to multiples of 64 */ 
  int ifm_pad = 0;
  int ofm_pad = 0;
  int nIfmp = nIfm;
  int nOfmp = nOfm;
  if (nIfm % 64 != 0){ 
      ifm_pad = (64-(nIfm%64));
      nIfmp += ifm_pad; 
  }
  if (nOfm % 64 != 0) {
      ofm_pad = (64-(nOfm%64));
      nOfmp += ofm_pad;  
  }

  gemm_conv_t gemm_param;
  /* set struct for naive convolution */
  gemm_param.nImg = nImg;
  gemm_param.nBIfm = nIfmp/CHANNEL_BLOCK;
  gemm_param.nbIfm = CHANNEL_BLOCK;
  gemm_param.nBOfm = nOfmp/CHANNEL_BLOCK;
  gemm_param.nbOfm = CHANNEL_BLOCK;
  gemm_param.ifhp = ifhp;
  gemm_param.ifwp = ifwp;
  gemm_param.ofhp = ofhp;
  gemm_param.ofwp = ofwp;
  gemm_param.ifh = ifh;
  gemm_param.ifw = ifw;
  gemm_param.ofh = ofh;
  gemm_param.ofw = ofw;
  gemm_param.pad_h = pad_h;
  gemm_param.pad_w = pad_w;

  if ( ofw == 56 ) {
    gemm_param.nbofw = 28;
  } else {
    gemm_param.nbofw = ofw;
  }

  gemm_param.kh = kh;
  gemm_param.kw = kw;
  gemm_param.stride_h = stride_h;
  gemm_param.stride_w = stride_w;
  gemm_input            = (float*)_mm_malloc( nImg*nIfmp*ifhp*ifwp*sizeof(float), 2097152);
  gemm_filter           = (float*)_mm_malloc( nOfmp*nIfmp*kh*kw*    sizeof(float), 2097152);
  gemm_filter_tr        = (float*)_mm_malloc( nOfmp*nIfmp*kh*kw*    sizeof(float), 2097152);
  gemm_output           = (float*)_mm_malloc( nImg*nOfmp*ofhp*ofwp*sizeof(float), 2097152);

  zero_buf(gemm_input, nImg*nIfmp*ifhp*ifwp);
  if (nIfm % 64 != 0 || nOfm % 64 != 0){ 
     zero_buf(gemm_filter, nOfmp*nIfmp*kh*kw);
     zero_buf(gemm_output, nImg*nOfmp*ofhp*ofwp);
  }

  /* copy data into GEMM optimized format */
  copy_pad_NCHW_to_GEMM(outputs, gemm_output, nImg, ofh, ofw, nOfm, CHANNEL_BLOCK, pad_h, pad_w, ofm_pad);
  copy_pad_KCRS_to_GEMM(weights, gemm_filter, kh, kw, nIfm, nOfm, CHANNEL_BLOCK, CHANNEL_BLOCK, ifm_pad, ofm_pad);
  gemm_kernel_conv_bp(&gemm_param, gemm_input, gemm_output, gemm_filter, gemm_filter_tr);
  /* copy out data */
  copy_pad_GEMM_to_NCHW(gemm_input, inputs, nImg, ifh, ifw, nIfm, CHANNEL_BLOCK, pad_h, pad_w, ifm_pad);

  _mm_free(gemm_input);
  _mm_free(gemm_filter);
  _mm_free(gemm_filter_tr);
  _mm_free(gemm_output);
  return 0;
}

int simple_conv2d_impl_wu(float *weights, float *outputs, float *inputs, int N, int C, int iH, int iW, 
	int K, int R, int S, int stride, int padding, int dilation, int groups) 
{
  float *gemm_input, *gemm_input_tr, *gemm_output, *gemm_filter;
  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w;

  int ifw = iW;         /* input width, "W" */
  int ifh = iH;         /* input height, "H" */
  int nImg = N;         /* mini-batch size, "N" */
  int nIfm = C;         /* number of input feature maps, "C" */
  int nOfm = K;         /* number of output feature maps, "K" */
  int kh = R;           /* filter height, "R" */
  int kw = S;           /* filter width, "S" */
  pad_h = padding;      /* padding in output */
  pad_w = padding;      /* padding in output */

  /* apply stride in both dimensions */
  stride_w = stride;
  stride_h = stride;

  /* deriving some values image size */
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  //ofw = (ifw + 2 * pad_w - dilation * (kw-1) - 1) / stride_w + 1;
  //ofh = (ifh + 2 * pad_h - dilation * (kh-1) - 1) / stride_h + 1;

  /* padding feature map width to be a multiple of 4 to perform VNN4 operations */
  int pad_iw = 0;
  int pad_ow = 0;

  if (ofw%4 != 0){ 
      pad_ow = 4-(ofw%4);
  }
  if (ifw%4 != 0){ 
      pad_iw = 4-(ifw%4);
  }
  pad_iw = stride_w * pad_ow;
  
  /*pad ofw and ifw to to be multiples of 4 (VNNI4) */ 
  ifhp = ifh + 2 * pad_h;
  ifwp = ifw + 2 * pad_w + pad_iw;
  ofhp = ofh + 2 * pad_h;
  ofwp = ofw + 2 * pad_w + pad_ow;

  /*pad nIfm and nOfm to multiples of 64 */ 
  int ifm_pad = 0;
  int ofm_pad = 0;
  int nIfmp = nIfm;
  int nOfmp = nOfm;
  if (nIfm % 64 != 0){ 
      ifm_pad = (64-(nIfm%64));
      nIfmp += ifm_pad; 
  }
  if (nOfm % 64 != 0) {
      ofm_pad = (64-(nOfm%64));
      nOfmp += ofm_pad;  
  }

  gemm_conv_t gemm_param;
  /* set struct for naive convolution */
  gemm_param.nImg = nImg;
  gemm_param.nBIfm = nIfmp/CHANNEL_BLOCK;
  gemm_param.nbIfm = CHANNEL_BLOCK;
  gemm_param.nBOfm = nOfmp/CHANNEL_BLOCK;
  gemm_param.nbOfm = CHANNEL_BLOCK;
  gemm_param.ifhp = ifhp;
  gemm_param.ifwp = ifwp;
  gemm_param.ofhp = ofhp;
  gemm_param.ofwp = ofwp;
  gemm_param.ifh = ifh;
  gemm_param.ifw = ifw;
  gemm_param.ofh = ofh;
  gemm_param.ofw = ofw;
  gemm_param.pad_h = pad_h;
  gemm_param.pad_w = pad_w;
  gemm_param.pad_iw = pad_iw;
  gemm_param.pad_ow = pad_ow;

  if ( ofw == 56 ) {
    gemm_param.nbofw = 28;
  } else {
    gemm_param.nbofw = ofw;
  }

  gemm_param.kh = kh;
  gemm_param.kw = kw;
  gemm_param.stride_h = stride_h;
  gemm_param.stride_w = stride_w;
  gemm_input            = (float*)_mm_malloc( nImg*nIfmp*ifhp*ifwp*sizeof(float), 2097152);
  gemm_input_tr         = (float*)_mm_malloc( nImg*nIfmp*(ifhp+(2*pad_h))*(ifwp+(2*pad_w))*sizeof(float), 2097152);
  gemm_filter           = (float*)_mm_malloc( nOfmp*nIfmp*kh*kw*    sizeof(float), 2097152);
  gemm_output           = (float*)_mm_malloc( nImg*nOfmp*ofhp*ofwp*sizeof(float), 2097152);
  zero_buf(gemm_input, nImg*nIfmp*ifhp*ifwp);
  zero_buf(gemm_input_tr, nImg*(ifhp+(2*pad_h))*nIfmp*(ifwp+(2*pad_w)));
  zero_buf(gemm_output, nImg*nOfmp*ofhp*ofwp);
  zero_buf(gemm_filter, nOfmp*nIfmp*kh*kw);
  /* copy data into GEMM optimized format */
  /* compensate for the VNNI4 padding */
  copy_pad_NCHW_to_GEMM_ex(inputs, gemm_input, nImg, ifh, ifw, nIfm, CHANNEL_BLOCK, pad_h, pad_w, pad_iw, ifm_pad);
  copy_pad_NCHW_to_GEMM_ex(outputs, gemm_output, nImg, ofh, ofw, nOfm, CHANNEL_BLOCK, pad_h, pad_w, pad_ow , ofm_pad);
  gemm_kernel_conv_wu(&gemm_param, gemm_filter, gemm_output, gemm_input, gemm_input_tr);
  copy_pad_GEMM_to_KCRS(gemm_filter, weights, kh, kw, nIfm, nOfm, CHANNEL_BLOCK, CHANNEL_BLOCK, ifm_pad, ofm_pad);
  _mm_free(gemm_input);
  _mm_free(gemm_input_tr);
  _mm_free(gemm_filter);
  _mm_free(gemm_output);
  return 0;
}
