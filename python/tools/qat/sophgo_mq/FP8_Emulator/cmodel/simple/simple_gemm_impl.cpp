/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <stdio.h>
#include <immintrin.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define SCRATCH_SIZE 4294967296
int lazy_init = 1;
float* myscratch = NULL;

extern void MMEngine_avx2_ps(int m, int n, int k, float alpha, float *A, int lda,
                float *B, int ldb, float beta, float *C, int ldc);

__extern_always_inline
void copy_matrix_and_pad(const float* in, float* out, const int LD, const int OD, int ld_pad, int od_pad)
{
  int LDP, ODP;
  LDP = LD + ld_pad;
  ODP = OD + od_pad;
  int ld, od;
  int lp, op;

  for ( op = 0; op < OD; op++ ) {
    for ( lp = LD-1; lp < LDP; lp++ ) {
       out[op*LDP+lp] = 0.0;
    }
  }
  for ( op = OD-1; op < ODP; op++ ) {
    for ( lp = 0; lp < LDP; lp++ ) {
       out[op*LDP+lp] = 0.0;
    }
  }

#if defined(_OPENMP)
#pragma omp parallel for private(ld, od)
#endif
  for ( od = 0; od < OD; od++ ) {
    for ( ld = 0; ld < LD; ld++ ) {
       out[od*LDP+ld] = in[od*LD+ld];
    }
  }
}

__extern_always_inline
void copy_matrix_and_strip_pading(const float* in, float* out, const int LD, const int OD, int ld_pad, int od_pad)
{
  int LDP;
  LDP = LD + ld_pad;
  int ld, od;
#if defined(_OPENMP)
#pragma omp parallel for private(ld, od)
#endif
  for ( od = 0; od < OD; od++ ) {
    for ( ld = 0; ld < LD; ld++ ) {
       out[od*LD+ld] = in[od*LDP+ld];
    }
  }
}

int simple_sgemm_impl( char* transa, char* transb, int m, int n, int k,
		float alpha, float* a, int lda, float* b, int ldb,
		float beta, float* c, int ldc ) {
  float* myA = NULL;
  float* myB = NULL;
  float* myC = NULL;
  size_t ptlda = (size_t)(lda);
  size_t ptldb = (size_t)(ldb);
  size_t mylda = (size_t)(lda);
  size_t myldb = (size_t)(ldb);
  size_t myldc = (size_t)(ldc);
  int mym   = (size_t)(m);
  int myn   = (size_t)(n);
  int myk   = (size_t)(k);
  int m_pad  = 0;
  int n_pad  = 0;
  int k_pad  = 0;

  int o,p,q,pp,oo;

  /* check for size matching our TMUL emulation */
  if ( mym % 16 != 0 ) {
    m_pad = (16-(mym%16));
    mym += m_pad;
  }
  if ( myk % 64 != 0 ) {
    k_pad = (64-(myk%64));
    myk += k_pad;
  }
  if ( myn % 16 != 0 ) {
    n_pad = (16-(myn%16));
    myn += n_pad;
  }
  /* update leading dimensions with padded values */
  mylda = mym;
  myldb = myk;
  myldc = mym;

  /* lazy init of fp8_gemm state */
  if (lazy_init != 0) {
    lazy_init = 0;
    myscratch = (float*) _mm_malloc( SCRATCH_SIZE*sizeof(float), 4096 );
  }
  /* check for sufficient scratch size */
  if ( (*transa == 'N') && (*transb == 'N') ) {
    if ( ((ptlda*myk)+(ptldb*myn)) > SCRATCH_SIZE ) {
      return -1;
    }
  } else if ( (*transa == 'T') && (*transb == 'N')  ) {
    if ( ((ptlda*mym)+(ptldb*myn)) > SCRATCH_SIZE ) {
      return -2;
    }
    mylda = mym;
  } else if ( (*transa == 'N') && (*transb == 'T')  ) {
    if ( ((ptlda*myk)+(ptldb*myk)) > SCRATCH_SIZE ) {
      return -3;
    }
    myldb = myk;
  } else if ( (*transa == 'T') && (*transb == 'T')  ) {
    if ( ((ptlda*mym)+(ptldb*myk)) > SCRATCH_SIZE ) {
      return -4;
    }
    mylda = mym;
    myldb = myk;
  } else {
    assert((0 && "Error : Invalid parameters"));
    return -5;
  }

  /* set temp A and B pointers */
  myA = myscratch;
  myB = myscratch + (mylda*myk);
  myC = myscratch + (mylda*myk) + (myldb*myn);

  if ( *transa == 'T' ) {
    /* fill the padding with zeros */
    for ( p = 0; p < k; p++ ) {
      for ( o = m-1; o < mym; o++ ) {
        myA[p*mylda+o] = 0.0;
      }
    }
    for ( p = k-1; p < myk; p++ ) {
      for ( o = 0; o < mym; o++ ) {
        myA[p*mylda+o] = 0.0;
      }
    }

    /* let's transpose data first */
#if defined(_OPENMP)
#pragma omp parallel for private(o,p) collapse(2)
#endif
    for ( p = 0; p < k; p++ ) {
      for ( o = 0; o < m; o++ ) {
        myA[(p*mylda)+o] = a[(o*ptlda)+p];
      }
    }
  } else if ( m_pad > 0 || k_pad > 0 ) {
    copy_matrix_and_pad(a, myA, m, k, m_pad, k_pad);
  } else {
    myA = a;
  }

  if ( *transb == 'T' ) {
    /* fill the padding with zeros */
    for ( p = 0; p < n; p++ ) {
      for ( o = k-1; o < myk; o++ ) {
        myB[p*myldb+o] = 0.0;
      }
    }
    for ( p = n-1; p < myn; p++ ) {
      for ( o = 0; o < myk; o++ ) {
        myB[p*myldb+o] = 0.0;
      }
    }

    /* let's transpose data first */
#if defined(_OPENMP)
#pragma omp parallel for private(o,p) collapse(2)
#endif
    for ( p = 0; p < n; p++ ) {
      for ( o = 0; o < k; o++ ) {
        myB[(p*myldb)+o] = b[(o*ptldb)+p];
      }
    }
  } else if ( k_pad > 0 || n_pad > 0 ) {
    copy_matrix_and_pad(b, myB, k, n, k_pad, n_pad);
  } else {
    myB = b;
  }

  if ( m_pad > 0 || n_pad > 0 ) {
    copy_matrix_and_pad(c, myC, m, n, m_pad, n_pad);
  } else {
    myC = c;
  }
  /* run gemm */
#if defined(_OPENMP)
#pragma omp parallel for private(o,p,q,pp,oo) collapse(2)
#endif
   for ( o = 0; o < mym; o += 16 ) {
    for ( p = 0; p < myn; p += 16 ) {
      float ctmp[256];
      for ( pp = 0; pp < 16; pp++ ) {
        for ( oo = 0; oo < 16; oo++ ) {
          ctmp[(pp*16)+oo] = 0.0f;
        }
      }
      for ( q = 0; q < myk; q += 64 ) {
	MMEngine_avx2_ps(mym, myn, myk, alpha, &(myA[(mylda*q)+o]), mylda,
			         &(myB[(myldb*p)+q]), myldb, beta, ctmp, myldc);
      }
      for ( pp = 0; pp < 16; pp++ ) {
        for ( oo = 0; oo < 16; oo++ ) {
          myC[((p+pp)*myldc)+(o+oo)] += ((alpha)*ctmp[(pp*16)+oo]) + ((beta)*myC[((p+pp)*myldc)+(o+oo)]);
        }
      }
    }
  }
  if ( m_pad > 0 || n_pad > 0 ) {
    copy_matrix_and_strip_pading(myC, c, m, n, m_pad, n_pad);
  }
  return 0;
}
