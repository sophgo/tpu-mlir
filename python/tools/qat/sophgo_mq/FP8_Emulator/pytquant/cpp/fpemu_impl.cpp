/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <torch/extension.h>
#include <vector>
#include <immintrin.h>
#include <omp.h>

enum ROUNDING_MODES {
  ROUND_RTZ = 0,
  ROUND_RNE = 1,
  ROUND_STOCHASTIC = 2,
  ROUND_RNAZ = 3,
  ROUND_RNTZ = 4,
  ROUND_PINF = 5,
  ROUND_NINF = 6
};

namespace {

  typedef union half_t {
    unsigned short u;
      at::Half f;
  } __half_t;

  typedef union ufloat32 {
    unsigned u;
    float f;
  } __float_t;

/* Following implementation of xoroshiro128++ PRNG is borrowed from here:
    http://prng.di.unimi.it/xoshiro128plusplus.c
    main page: http://prng.di.unimi.it/
*/
  static uint32_t s1_[4] = { 1387366120, 2798441831, 888998500 , 1099633400 };
  static uint32_t s2_[4] = { 2034269327, 2125325156, 1209715489, 1931656721 };
  static uint32_t s3_[4] = { 1555452618, 650181557 , 883695203 , 627677842  };
  static uint32_t s4_[4] = { 4195248041, 2146478152, 480059239 , 1468956197 };
  static uint32_t s5_[4] = { 1252084877, 500390994 , 977516591 , 1950666000 };
  static uint32_t s6_[4] = { 3936597502, 834151069 , 1477014702, 734008143  };
  static uint32_t s7_[4] = { 1983400973, 1164103095, 2110188261, 2019272068 };
  static uint32_t s8_[4] = { 1877096364, 2833629967, 4196320416, 1774181187 };
  static uint32_t s9_[4] = { 702309618 , 4077815558, 1512057936, 1868769368 };
  static uint32_t s10_[4] =
    { 510001215 , 966559856 , 776583255 , 1475621065 };
  static uint32_t s11_[4] =
    { 1271806057, 1881312534, 478635452 , 814821902  };
  static uint32_t s12_[4] =
    { 733990058 , 1889991804, 1108257970, 1093480892 };
  static uint32_t s13_[4] =
    { 4273743809, 4167473370, 558000409 , 1594848927 };
  static uint32_t s14_[4] =
    { 444870959 , 1595722866, 1064124488, 3637102547 };
  static uint32_t s15_[4] =
    { 703721499 , 3896407831, 1002360059, 1427395742 };
  static uint32_t s16_[4] =
    { 1295231497, 1254972431, 1423497865, 861918264  };

/* seed pointer array */
  static uint32_t *sptr_[16] = { s1_, s2_, s3_, s4_, s5_, s6_, s7_, s8_, s9_,
    s10_, s11_, s12_, s13_, s14_, s15_, s16_
  };

  static inline uint32_t rotl (const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
  }

  uint32_t rand_xorshft128plus_scalar (uint32_t * ps) {
    const uint32_t result_plus = ps[0] + ps[3];
    const uint32_t t = ps[1] << 9;

    ps[2] ^= ps[0];
    ps[3] ^= ps[1];
    ps[1] ^= ps[2];
    ps[0] ^= ps[3];

    ps[2] ^= t;

    ps[3] = rotl (ps[3], 11);

    return result_plus;
  }

  inline __m512i
    _mm512_rndxorshft128plus_epi32 (uint32_t * vs0, uint32_t * vs1,
				    uint32_t * vs2, uint32_t * vs3) {
    __m512i vrplus = _mm512_add_epi32 (_mm512_load_epi32 (vs0), _mm512_load_epi32 (vs3));
    __m512i vt = _mm512_sll_epi32 (_mm512_load_epi32 (vs1), _mm_set1_epi8 (9));
    _mm512_store_epi32 (vs2, _mm512_xor_epi32 (_mm512_load_epi32 (vs2), _mm512_load_epi32 (vs0)));
    _mm512_store_epi32 (vs3, _mm512_xor_epi32 (_mm512_load_epi32 (vs3), _mm512_load_epi32 (vs1)));
    _mm512_store_epi32 (vs1, _mm512_xor_epi32 (_mm512_load_epi32 (vs1), _mm512_load_epi32 (vs2)));
    _mm512_store_epi32 (vs0, _mm512_xor_epi32 (_mm512_load_epi32 (vs0), _mm512_load_epi32 (vs3)));
    _mm512_store_epi32 (vs2, _mm512_xor_epi32 (_mm512_load_epi32 (vs2), vt));

    __m512i vl = _mm512_sll_epi32 (_mm512_load_epi32 (vs3), _mm_set1_epi8 (11));
    __m512i vr = _mm512_sra_epi32 (_mm512_load_epi32 (vs3), _mm_set1_epi8 (21));
    _mm512_store_epi32 (vs3, _mm512_or_epi32 (vl, vr));
    return vrplus;
  }


  float __double2float_rn (double inval) {
    float out[4] = { 0 };
    __m128 vout = _mm_cvtpd_ps (_mm_set1_pd (inval));

    _mm_store_ps (&out[0], vout);
    return out[0];
  }

  unsigned short __float2half_rn (float inval) {
    return _cvtss_sh (inval, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }

  float __half2float (unsigned short h_val) {
    return _cvtsh_ss (h_val);
  }

  template < typename scalar_t > float __anyfloat2float_rn (scalar_t a_) {
    float f_;

    if (std::is_same < scalar_t, double >::value) {
      f_ = __double2float_rn (a_);
    } else if (std::is_same < scalar_t, float >::value) {
      f_ = a_;
    } else if (std::is_same < scalar_t, at::Half >::value) {
      f_ = __half2float ((at::Half) a_);
    }
    return f_;
  }

  template < typename scalar_t >
    void __float2anyfloat_rn (float f_, scalar_t * out) {
    scalar_t a_;

    if (std::is_same < scalar_t, double >::value) {
      a_ = (scalar_t) (f_);
    } else if (std::is_same < scalar_t, float >::value) {
      a_ = f_;
    } else if (std::is_same < scalar_t, at::Half >::value) {
      a_ = (at::Half) __float2half_rn (f_);
    }
    *out = a_;
  }

  template < typename scalar_t >
    unsigned short __anyfloat2half_rn (scalar_t f_) {
    unsigned short h_;

    if (std::is_same < scalar_t, double >::value) {
      h_ = __float2half_rn (__double2float_rn (f_));
    } else if (std::is_same < scalar_t, float >::value) {
      h_ = __float2half_rn (f_);
    } else if (std::is_same < scalar_t, at::Half >::value) {
      unsigned short *ptrh_ = (unsigned short *) &f_;
      h_ = *ptrh_;
    }
    return h_;
  }

  template < typename scalar_t >
    void __half2anyfloat (unsigned short h_, scalar_t * out, scalar_t scale=1.0) {
    scalar_t f_;

    if (std::is_same < scalar_t, double >::value) {
      f_ = (scalar_t) __half2float (h_);
    } else if (std::is_same < scalar_t, float >::value) {
      f_ = __half2float (h_);
    } else if (std::is_same < scalar_t, at::Half >::value) {
      f_ = *((at::Half *) & h_);
    }
    *out = scale*f_;
  }

  template < typename scalar_t >
    inline void reduce0 (scalar_t * g_data, float *g_odata, unsigned int n) {
    float sum = 0.0, sumsq = 0.0;

#pragma omp parallel for reduction(+: sum), reduction(+: sumsq)
    for (unsigned int i = 0; i < n; i++) {
      sum += __anyfloat2float_rn (g_data[i]);
      sumsq += __anyfloat2float_rn (g_data[i]) * __anyfloat2float_rn (g_data[i]);
    }
    g_odata[0] = sum;
    g_odata[1] = sumsq;
  }

  template < typename scalar_t >
    inline void absmax0 (scalar_t * g_data, float *g_odata, unsigned int n) {
    float absmax = 0.0;

#pragma omp parallel for reduction(max: absmax)
    for (unsigned int i = 0; i < n; i++) {
      absmax = fmaxf (absmax, fabsf (__anyfloat2float_rn (g_data[i])));
    }
    g_odata[0] = absmax;
  }


  void cvt_fp32_bf16_rne_intrinsic (const float *__restrict__ in, float *out,
				    int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i += 16) {
      const __m512i vnaninf = _mm512_set1_epi32 (0x7f800000), vrneadd =	_mm512_set1_epi32 (0x00007fff);
      const __m512i vfixup = _mm512_set1_epi32 (0x00000001), vfixupmask = _mm512_set1_epi32 (0x00010000);
      const __m512i truncmask = _mm512_set1_epi32 (0xffff0000);
      __m512 a = _mm512_loadu_ps (&in[i]);
      const __m512i mm512_roundbf16rne_a_ = _mm512_castps_si512 (a);
      const __mmask16 mm512_roundbf16rne_mask1_ =
	_mm512_cmp_epi32_mask (_mm512_and_epi32
			       (mm512_roundbf16rne_a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 mm512_roundbf16rne_mask2_ =
	_mm512_cmp_epi32_mask (_mm512_and_epi32
			       (mm512_roundbf16rne_a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      __m512i roundbf16rne =
	_mm512_mask_add_epi32 (mm512_roundbf16rne_a_,
			       mm512_roundbf16rne_mask1_,
			       mm512_roundbf16rne_a_,
			       _mm512_mask_add_epi32 (vrneadd,
						      mm512_roundbf16rne_mask2_, vrneadd, vfixup));
      a = _mm512_castsi512_ps (_mm512_and_epi32 (roundbf16rne, truncmask));

      _mm512_storeu_ps (&out[i], a);
    }
  }

  void cvt_fp32_bf16_stochastic_intrinsic (const float *__restrict__ in,
					   float *out, int size) {
    uint32_t vs0[16] __attribute__ ((aligned (64))) = {
    1387366120, 279844183, 888998500, 1099633400, 1252084877, 500390994,
	977516591, 1950666000, 393659750, 834151069, 1477014702, 734008143,
	1983400973, 116410309, 2110188261, 2019272068};
    uint32_t vs1[16] __attribute__ ((aligned (64))) = {
    2034269327, 2125325156, 1209715489, 193165672, 187709636, 28336299,
	419632041, 1774181187, 702309618, 407781555, 1512057936, 1868769368,
	510001215, 966559856, 776583255, 147562106};
    uint32_t vs2[16] __attribute__ ((aligned (64))) = {
    1555452618, 650181557, 883695203, 62767784, 127180605, 1881312534,
	478635452, 814821902, 733990058, 1889991804, 1108257970, 1093480892,
	427374380, 416747337, 558000409, 1594848927};
    uint32_t vs3[16] __attribute__ ((aligned (64))) = {
    419524804, 2146478152, 480059239, 1468956197, 444870959, 1595722866,
	1064124488, 363710254, 703721499, 389640783, 1002360059, 1427395742,
	1295231497, 1254972431, 1423497865, 861918264};

#pragma omp parallel for firstprivate (vs0, vs1, vs2, vs3)
    for (int i = 0; i < size; i += 16) {
      const __m512i vnaninf = _mm512_set1_epi32 (0x7f800000), vrneadd =	_mm512_set1_epi32 (0x00007fff);
      const __m512i vfixup = _mm512_set1_epi32 (0x00000001), vfixupmask = _mm512_set1_epi32 (0x00010000);
      const __m512i truncmask = _mm512_set1_epi32 (0xffff0000);
      __m512i rnd512 = _mm512_rndxorshft128plus_epi32 (vs0, vs1, vs2, vs3);
      __m256i rnbits = _mm512_extracti32x8_epi32 (rnd512, 0);

      __m512 a = _mm512_loadu_ps (&in[i]);
      const __m512i mm512_roundbf16sr_a_ = _mm512_castps_si512 (a);
      const __mmask16 mm512_roundbf16rne_mask1_ =
	_mm512_cmp_epi32_mask (_mm512_and_epi32(mm512_roundbf16sr_a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 mm512_roundbf16rne_mask2_ =
	_mm512_cmp_epi32_mask (_mm512_and_epi32(mm512_roundbf16sr_a_, vfixupmask), vfixupmask,_MM_CMPINT_EQ);
      __m512i roundbf16sr =
	_mm512_mask_add_epi32 (mm512_roundbf16sr_a_,
			       mm512_roundbf16rne_mask1_,
			       mm512_roundbf16sr_a_,
			       _mm512_cvtepu16_epi32 (rnbits));
      roundbf16sr =
	_mm512_mask_add_epi32 (roundbf16sr, mm512_roundbf16rne_mask1_,
			       roundbf16sr, _mm512_mask_add_epi32 (vrneadd,
								   mm512_roundbf16rne_mask2_,
								   vrneadd,
								   vfixup));
      a = _mm512_castsi512_ps (_mm512_and_epi32 (roundbf16sr, truncmask));

      _mm512_storeu_ps (&out[i], a);
    }
  }

  void cvt_fp32_bf16_scalar (const float *in, float *out, const int size,
			     int rmode) {
    int lshift = 16;
    int rshift = lshift - 3;	/* shift to preserve rounding bits */
    unsigned int mask_mant = (unsigned int) (0xFFFFFFFF << lshift);
    unsigned int mask_mant_grs = (unsigned int) (0xFFFFFFFF << rshift);

    /* mask to extract G(gaurd), R (round), S (sticky) bits */
    unsigned int lsbGRS = 0xF << rshift;

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */

    if (rmode == ROUND_RNE)
      rne_mask = 1;
    if (rmode == ROUND_STOCHASTIC)
      sr_mask = 1;

    for (int gid = 0; gid < size; gid++) {
      __float_t uf;

      uf.f = in[gid];
      unsigned int mant_grs = (uf.u & mask_mant_grs);

      if (sr_mask) {
	/* stochastic with 16 seeds */
	int seed_index = (gid / 16);
	unsigned short rand =
	  (unsigned short)
	  rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	/* stochastic rounding with 16-bit random number */
	uf.u += rand;
      }
      /* truncation */
      uf.u &= mask_mant;

      /* round to nearest even after truncation if rne_mask is enabled */
      unsigned int rmask_tie = ((mant_grs & lsbGRS) >> rshift);
      unsigned int rmask = (rmask_tie & 0x7);

      uf.u += rne_mask * (((rmask > 0x4) || (rmask_tie == 0xC)) << lshift);

      //__float2anyfloat_rn(uf.f, &out[gid]);
      out[gid] = uf.f;
    }
  }

  template < typename scalar_t >
    void BFLOAT16_Kernel (const scalar_t * in,
				 scalar_t * out, const int size, int rmode) {
    if ((size % 16) == 0) {
      if (rmode == ROUND_STOCHASTIC)
	cvt_fp32_bf16_stochastic_intrinsic (in, out, size);
      else
	cvt_fp32_bf16_rne_intrinsic (in, out, size);
    } else {
      int vec_size = ((int) (size / 16)) * 16;

      if (vec_size > 0) {
	if (rmode == ROUND_STOCHASTIC)
	  cvt_fp32_bf16_stochastic_intrinsic (in, out, vec_size);
	else
	  cvt_fp32_bf16_rne_intrinsic (in, out, vec_size);
      }
      cvt_fp32_bf16_scalar (&in[vec_size], &out[vec_size], size - vec_size, rmode);
    }
  }

  template < typename scalar_t >
    void FLOAT16_Kernel (const scalar_t * in,
				scalar_t * out,
				const int size, int rmode, int no_denorm) {
    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */

    if (rmode == ROUND_RNE)
      rne_mask = 1;
    if (rmode == ROUND_STOCHASTIC)
      sr_mask = 1;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      if (rne_mask) {
	__half_t h;

	h.u = __anyfloat2half_rn (in[gid]);
	unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0);
	unsigned short is_denorm = (not_denorm == 0) ? 1 : 0;

	h.u *= !(is_denorm && no_denorm);
	__half2anyfloat (h.u, &out[gid]);
      } else if (sr_mask) {
	unsigned int fval = ((unsigned int *) in)[gid];
	int exp_h = (int) ((fval & 0x7f800000) >> 23) - 127;
	unsigned int mantissa_h = (fval & 0x7FFFFF);
	unsigned int sign_h = (fval & 0x80000000);
	__half_t h;

	if (exp_h == 128) {
	  /* handle incoming INF and NaN */
	  exp_h = 0x1F;
	  /* handle signalling NaN */
	  if (mantissa_h && ((mantissa_h & 0x400000) == 0x0))
	    mantissa_h |= 0x400000;
	  mantissa_h |= (exp_h << 23);
	  mantissa_h |= (sign_h >> 3);
	  h.u = (unsigned short) (mantissa_h >> 13);
	} else if (exp_h >= 16) {
	  /* saturate to INF */
	  exp_h = 0x1F;
	  mantissa_h = 0;
	  mantissa_h |= (exp_h << 23);
	  mantissa_h |= (sign_h >> 3);
	  h.u = (unsigned short) (mantissa_h >> 13);
	} else {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned int rand =
	    (unsigned int)
	    rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  if (exp_h < -14) {
	    mantissa_h += (rand & 0x00001FFF);
	    /* handle denormals */
	    h.u = __anyfloat2half_rn (in[gid]);
	  } else {
	    exp_h += 15;
	    mantissa_h |= (exp_h << 23);
	    mantissa_h |= (sign_h >> 3);
	    mantissa_h += (rand & 0x00001FFF);
	    h.u = (unsigned short) (mantissa_h >> 13);
	  }
	}
	__half2anyfloat (h.u, &out[gid]);
      }
    }
  }


  __m256i _mm256_cvt2fp16_e5m2 (__m256i a, __m256i b) {
    const __m512i vnaninf = _mm512_set1_epi16 (0x7c00), vrneadd =
      _mm512_set1_epi16 (0x007f);
    const __m512i vfixup = _mm512_set1_epi16 (0x0001), vfixupmask =
      _mm512_set1_epi16 (0x0100);
    /* b: lower half, a : upper half */
    const __m512i a_ =
      _mm512_inserti64x4 (_mm512_inserti64x4 (_mm512_setzero_si512 (), b, 0), a, 1);
    const __mmask32 maska1_ =
      _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vnaninf), vnaninf,_MM_CMPINT_NE);
    const __mmask32 maska2_ =
      _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vfixupmask), vfixupmask,_MM_CMPINT_EQ);
    __m512i a_rne_ = _mm512_mask_add_epi16 (a_, maska1_, a_,
			     _mm512_mask_add_epi16 (vrneadd, maska2_, vrneadd,vfixup));
    return _mm512_cvtepi16_epi8 (_mm512_srli_epi16 (a_rne_, 8));
  }

  static inline __m512i _mm512_cvte5m2_fp16 (__m256i a) {
    return _mm512_slli_epi16 (_mm512_cvtepi8_epi16 (a), 8);
  }

  static inline void cvt_fp16_e5m2_rne_intrinsic (const short *__restrict__ in,
				   unsigned char *out, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i += 32) {
      __m256i bh_ = _mm256_lddqu_si256 ((__m256i *) & in[i]);
      __m256i ah_ = _mm256_lddqu_si256 ((__m256i *) & in[i + 16]);

      _mm256_storeu_si256 ((__m256i *) & out[i],
			   _mm256_cvt2fp16_e5m2 (ah_, bh_));
    }
  }

  __m256i _mm256_cvt2fp16_e5m2_noINF (__m256i a, __m256i b) {
    const __m512i vnaninf    = _mm512_set1_epi16 (0x7c00);
    const __m512i vrneadd    = _mm512_set1_epi16 (0x007f);
    const __m512i vfixup     = _mm512_set1_epi16 (0x0001);
    const __m512i vfixupmask = _mm512_set1_epi16 (0x0100);
    /* use a non-standard exponent offset = 16, */
    const __m512i vExp_fp16  = _mm512_set1_epi16 (0x000F);
    const __m512i vExp_e5m2  = _mm512_set1_epi16 (0x0010);
    const __m512i vsMant     = _mm512_set1_epi16 (0x83FF);
    /* Exponent Offset = 16, reclaim inf/NaN */
    const __m512i vsatuval   = _mm512_set1_epi16 (0x7F00);/* 2^15*1.11 a.k.a 57344.0, largest value */
    const __m512i vinfval    = _mm512_set1_epi16 (0x8000);	/* -0.0 as INF */
    const __m512i a_ = _mm512_inserti64x4 (_mm512_inserti64x4 (_mm512_setzero_si512 (), b, 0), a, 1);
    const __mmask32 maska1_ = _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
    const __mmask32 maska2_ = _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
    const __mmask32 maska3_ = _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, _mm512_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLE);

    __m512i vExp_ = _mm512_sub_epi16 (_mm512_srli_epi16 (_mm512_and_si512 (a_, vnaninf), 10), vExp_fp16);
    vExp_ = _mm512_slli_epi16 (_mm512_add_epi16 (vExp_, vExp_e5m2), 10);
    __m512i a_rne_ = _mm512_or_si512 (vExp_, _mm512_and_si512 (a_, vsMant));

    a_rne_ = _mm512_mask_add_epi16 (a_rne_, maska1_, a_rne_,
			     _mm512_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
    a_rne_ = _mm512_mask_mov_epi16 (a_rne_, maska3_, _mm512_or_si512(_mm512_and_si512(a_rne_, vinfval), vsatuval));
    a_rne_ = _mm512_mask_mov_epi16 (a_rne_, ~maska1_, vinfval);
    return _mm512_cvtepi16_epi8 (_mm512_srli_epi16 (a_rne_, 8));
  }

  static inline __m256i _mm512_cvt2fp32_e5m2_noINF (__m512 a, __m512 b) {
    __m256i ah_ = _mm512_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ = _mm512_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return _mm256_cvt2fp16_e5m2_noINF (ah_, bh_);
  }

  __m512i _mm512_cvte5m2_noinf_fp16 (__m256i a) {
    const __m512i vExp_fp16 = _mm512_set1_epi16 (0x000F);
    const __m512i vExp_e5m2 = _mm512_set1_epi16 (0x0010);
    const __m512i vsMant    = _mm512_set1_epi16 (0x83FF);
    const __m512i vnaninf   = _mm512_set1_epi16 (0x8000);	/* -0.0 as INF */
    const __m512i vinfval   = _mm512_set1_epi16 (0x7c00);
    __m512i a_ = _mm512_slli_epi16 (_mm512_cvtepi8_epi16 (a), 8);
    const __mmask32 mask1_ = _mm512_cmp_epi16_mask (a_, vnaninf, _MM_CMPINT_EQ);
    __m512i vExp_ = _mm512_sub_epi16 (_mm512_srli_epi16 (_mm512_and_si512 (a_, vinfval), 10), vExp_e5m2);
    vExp_ = _mm512_slli_epi16 (_mm512_add_epi16 (vExp_, vExp_fp16), 10);
    a_ = _mm512_or_si512 (vExp_, _mm512_and_si512 (a_, vsMant));
    return _mm512_mask_mov_epi16 (a_, mask1_, vinfval);
  }

  static inline void cvt_fp16_e5m2_noINF_rne_intrinsic (const short *__restrict__ in,
					 unsigned char *out, int size) {
#pragma omp parallel for
    for (int i = 0; i < size; i += 32) {
      __m256i bh_ = _mm256_lddqu_si256 ((__m256i *) & in[i]);
      __m256i ah_ = _mm256_lddqu_si256 ((__m256i *) & in[i + 16]);

      _mm256_storeu_si256 ((__m256i *) & out[i], _mm256_cvt2fp16_e5m2 (ah_, bh_));
    }
  }

  void cvt_fp32_e5m2_noinf_rne_intrinsic (const float *__restrict__ in,
					 float *out, int size, float scale) {
#pragma omp parallel for
    for (int i = 0; i < size; i += 32) {
      __m512 b = _mm512_loadu_ps (&in[i]);
      __m512 a = _mm512_loadu_ps (&in[i + 16]);
      __m256i ah_ = _mm512_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i bh_ = _mm512_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m512i a_rne_ = _mm512_cvte5m2_noinf_fp16 (_mm256_cvt2fp16_e5m2_noINF (ah_, bh_));
      bh_ = _mm512_extracti64x4_epi64 (a_rne_, 0);
      ah_ = _mm512_extracti64x4_epi64 (a_rne_, 1);
      b = _mm512_cvtph_ps (bh_);
      a = _mm512_cvtph_ps (ah_);
      _mm512_storeu_ps (&out[i], b);
      _mm512_storeu_ps (&out[i + 16], a);
    }
  }

  static inline void cvt_fp32_e5m2_flex_intrinsic (const float *__restrict__ in, float *out,
				    int size, float scale) {
#pragma omp parallel for
    for (int i = 0; i < size; i += 16) {
      const __m512i vnaninf    = _mm512_set1_epi32 (0x7f800000);
      const __m512i vrneadd    = _mm512_set1_epi32 (0x000fffff);
      const __m512i vfixup     = _mm512_set1_epi32 (0x00000001);
      const __m512i vfixupmask = _mm512_set1_epi32 (0x00200000);
      const __m512i vexpf32    = _mm512_set1_epi32 (0x0000007f);
      const __m512i vmexp_e5m2 = _mm512_set1_epi32 (0x0000000f);
      const __m512i vsign      = _mm512_set1_epi32 (0x80000000);
      const __m512i vmant      = _mm512_set1_epi32 (0x007fffff);
      const __m512i vmin_e5m2  = _mm512_set1_epi32 (0x37800000);
      /*const __m512i vmax_e5m2 = _mm512_set1_epi32 (0x47600000);*/
      const __m512i vdnorm     = _mm512_set1_epi32 (0x38800000);

      __m512 a = _mm512_loadu_ps (&in[i]);
      const __m512i a_ = _mm512_castps_si512 (a);
      const __mmask16 naninf_mask_ = _mm512_cmp_epi32_mask (_mm512_and_epi32 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 rneadd_mask_ = _mm512_cmp_epi32_mask (_mm512_and_epi32 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 zflush_mask_ = _mm512_cmp_epi32_mask (_mm512_and_epi32 (a_, vnaninf), vmin_e5m2, _MM_CMPINT_LT);
      const __mmask16 denorm_mask_ = _mm512_cmp_epi32_mask (_mm512_and_epi32 (a_, vnaninf), vdnorm, _MM_CMPINT_LT);

      __m512i a_sign = _mm512_and_epi32 (a_, vsign);
      __m512i a_rne = _mm512_mask_add_epi32 (a_, naninf_mask_, a_,
			       _mm512_mask_add_epi32 (vrneadd, rneadd_mask_, vrneadd, vfixup));
      __m512i a_exp = _mm512_srai_epi32 (_mm512_and_epi32 (a_rne, vnaninf), 23);
      a_rne = _mm512_and_epi32 (a_rne, vmant);
      a_exp = _mm512_sub_epi32 (a_exp, vexpf32);

      __m512i vminexp = _mm512_sub_epi32 (_mm512_setzero_epi32 (), vmexp_e5m2);
      __m512i shft_ = _mm512_sub_epi32 (vminexp, a_exp);

      /*const __mmask16 ovflow_mask_ =
	_mm512_cmp_epi32_mask (a_exp, vmexp_e5m2, _MM_CMPINT_NLT);*/
      a_exp = _mm512_add_epi32 (a_exp, vmexp_e5m2);

      __m512i vrshft = _mm512_set1_epi32 (21);

      vrshft = _mm512_mask_add_epi32 (vrshft, denorm_mask_, vrshft, shft_);
      __m512i vlshft = _mm512_set1_epi32 (8);

      vlshft = _mm512_mask_add_epi32 (vlshft, denorm_mask_, vlshft, shft_);

      a_rne = _mm512_sllv_epi32 (_mm512_srav_epi32 (a_rne, vrshft), vlshft);

      a_exp = _mm512_slli_epi32 (a_exp, 10);
      a_rne = _mm512_or_epi32 (a_rne, a_exp);
      a_rne = _mm512_or_epi32 (a_rne, _mm512_srai_epi32 (a_sign, 16));
      a_rne = _mm512_mask_set1_epi32 (a_rne, zflush_mask_, 0);

      __m256i a_rne_16 = _mm512_cvtepi32_epi16 (a_rne);

      a = _mm512_cvtph_ps (a_rne_16);
      _mm512_storeu_ps (&out[i], a);
    }
  }

  void cvt_fp32_e5m2_rne_intrinsic (const float *__restrict__ in, float *out,
				   int size, float scale) {

#pragma omp parallel for
    for (int i = 0; i < size; i += 32) {
      const __m512i vnaninf    = _mm512_set1_epi16 (0x7c00);
      const __m512i vrneadd    = _mm512_set1_epi16 (0x007f);
      const __m512i vfixup     = _mm512_set1_epi16 (0x0001);
      const __m512i vfixupmask = _mm512_set1_epi16 (0x0100);

      __m512 s_ = _mm512_set1_ps (scale);
      __m512 sr_ = _mm512_set1_ps (1.0 / scale);
      __m512 b = _mm512_loadu_ps (&in[i]);
      __m512 a = _mm512_loadu_ps (&in[i + 16]);

      b = _mm512_mul_ps (b, s_);
      a = _mm512_mul_ps (a, s_);

      __m256i ah_ = _mm512_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i bh_ = _mm512_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      const __m512i a_ = _mm512_inserti64x4 (_mm512_inserti64x4 (_mm512_setzero_si512 (), bh_, 0), ah_, 1);
      const __mmask32 maska1_ =	_mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask32 maska2_ =	_mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      __m512i a_rne_ = _mm512_mask_add_epi16 (a_, maska1_, a_,_mm512_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      a_rne_ = _mm512_slli_epi16 (_mm512_srli_epi16 (a_rne_, 8), 8);

      bh_ = _mm512_extracti64x4_epi64 (a_rne_, 0);
      ah_ = _mm512_extracti64x4_epi64 (a_rne_, 1);
      b = _mm512_cvtph_ps (bh_);
      a = _mm512_cvtph_ps (ah_);

      _mm512_storeu_ps (&out[i], _mm512_mul_ps (b, sr_));
      _mm512_storeu_ps (&out[i + 16], _mm512_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e5m2_stochastic_intrinsic (const float *__restrict__ in,
					  float *out, int size, float scale) {
    uint32_t vs0[16] __attribute__ ((aligned (64))) = {
        1387366120, 279844183, 888998500, 1099633400, 1252084877, 500390994,
	977516591, 1950666000, 393659750, 834151069, 1477014702, 734008143,
	1983400973, 116410309, 2110188261, 2019272068};
    uint32_t vs1[16] __attribute__ ((aligned (64))) = {
        2034269327, 2125325156, 1209715489, 193165672, 187709636, 28336299,
	419632041, 1774181187, 702309618, 407781555, 1512057936, 1868769368,
	510001215, 966559856, 776583255, 147562106};
    uint32_t vs2[16] __attribute__ ((aligned (64))) = {
        1555452618, 650181557, 883695203, 62767784, 127180605, 1881312534,
	478635452, 814821902, 733990058, 1889991804, 1108257970, 1093480892,
	427374380, 416747337, 558000409, 1594848927};
    uint32_t vs3[16] __attribute__ ((aligned (64))) = {
        419524804, 2146478152, 480059239, 1468956197, 444870959, 1595722866,
	1064124488, 363710254, 703721499, 389640783, 1002360059, 1427395742,
	1295231497, 1254972431, 1423497865, 861918264};

#pragma omp parallel for firstprivate (vs0, vs1, vs2, vs3)
    for (int i = 0; i < size; i += 32) {
      const __m512i vnaninf    = _mm512_set1_epi16 (0x7c00);
      const __m512i vfixup     = _mm512_set1_epi16 (0x0001);
      const __m512i vfixupmask = _mm512_set1_epi16 (0x0100);
      const __m512i vrneadd    = _mm512_set1_epi16 (0x007f);
      const __m512i vdenorm    = _mm512_set1_epi16 (0x03ff);
      const __m512i vexmant    = _mm512_set1_epi16 (0x7fff);

      __m512i rnd512 = _mm512_rndxorshft128plus_epi32 (vs0, vs1, vs2, vs3);
      __m256i rnbits = _mm512_extracti32x8_epi32 (rnd512, 0);

      __m512 s_ = _mm512_set1_ps (scale);
      __m512 sr_ = _mm512_set1_ps (1.0 / scale);

      __m512 b = _mm512_loadu_ps (&in[i]);
      __m512 a = _mm512_loadu_ps (&in[i + 16]);

      b = _mm512_mul_ps (b, s_);
      a = _mm512_mul_ps (a, s_);

      __m256i ah_ = _mm512_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i bh_ = _mm512_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      const __m512i a_ = _mm512_inserti64x4 (_mm512_inserti64x4 (_mm512_setzero_si512 (), bh_, 0), ah_, 1);
      const __mmask32 maska1_ = _mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask32 maska2_ =	_mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask32 maska4_ =	_mm512_cmp_epi16_mask (_mm512_and_si512 (a_, vexmant), vdenorm, _MM_CMPINT_LE);
      __m512i a_sr_ = _mm512_mask_add_epi16 (a_, (maska1_ & ~maska4_), a_, _mm512_cvtepu8_epi16 (rnbits));
      a_sr_ = _mm512_mask_add_epi16 (a_sr_, maska4_, a_sr_, _mm512_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      a_sr_ = _mm512_slli_epi16 (_mm512_srli_epi16 (a_sr_, 8), 8);

      bh_ = _mm512_extracti64x4_epi64 (a_sr_, 0);
      ah_ = _mm512_extracti64x4_epi64 (a_sr_, 1);
      b = _mm512_cvtph_ps (bh_);
      a = _mm512_cvtph_ps (ah_);

      _mm512_storeu_ps (&out[i], _mm512_mul_ps (b, sr_));
      _mm512_storeu_ps (&out[i + 16], _mm512_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e5m2_scalar (const float *__restrict__ in, float *out,
			    int size, float scale, int rmode) {
    int non_mant_bits = 5 /*exp_bits */  + 1;	/* exponent + sign */
    int lshift = 10 - (8 /*mbits */  - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x00FF;
    unsigned short rne_tie = 0x0180;

    float scale_reciprocal = 1.0 / scale;

    for (int gid = 0; gid < size; gid++) {
      __half_t h;
      float inval = scale * in[gid];

      h.u = __anyfloat2half_rn (inval);

      unsigned short can_round = ((h.u & 0x7F00) <= 0x7B00) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      /* nearest rounding masks */
      unsigned short rnmask = (h.u & grs_bitmask);
      unsigned short rnmask_tie = (h.u & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short)
	    rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  h.u += can_round * is_normal * (rand & 0xFF);
	  /* stochastic round:  denormals --> rne rounding */
	  h.u += can_round * is_denorm *
	    (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  h.u += can_round * rne_mask *
	    (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  h.u += can_round * rnaz_mask * ((rnmask >= 0x0080) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  h.u += can_round * rntz_mask * ((rnmask > 0x0080) << lshift);
	  /* round to +INF, if rpinf_mask is enabled */
	  h.u += can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0080) << lshift);
	  /* round to -INF, if rminf_mask is enabled */
	  h.u += can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0080) << lshift);
	}
      }
      /* truncation */
      h.u = (h.u & mask_mant);
      float f_;
      __half2anyfloat (h.u, &f_);
      out[gid] = f_ * scale_reciprocal;
    }
  }

  template < typename scalar_t >
    void E5M2_Kernel (const scalar_t * __restrict__ in,
		      scalar_t * __restrict__ out,
		      const int size,
		      const scalar_t in_scale,
		      bool block_norm, int block_size, int rmode) {
    float scale = in_scale;
    float fmax = std::numeric_limits<scalar_t>::max();
    if (scale > fmax) {
      fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
      exit(1);
    }

    if (block_norm == true) {
      int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
      for (int b = 0; b < nblocks; b++) {
	int start_index = (b * block_size);
	/* handle the last block */
	if (start_index + block_size > size)
	  block_size = (size - start_index);

	float maxval = 0.0;

#pragma omp parallel for reduction (max:maxval)
	for (int gid = start_index; gid < start_index + block_size; gid++) {
	  maxval = (maxval < fabs (in[gid])) ? fabs (in[gid]) : maxval;
	}
	__float_t f;

	f.f = maxval;
	f.u = (f.u & 0x7F800000);
	scale = 2.0 * f.f;
	scale /= 16384.0;

	if ((block_size % 32) == 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e5m2_stochastic_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  } else {
	    cvt_fp32_e5m2_rne_intrinsic (&in[start_index], &out[start_index], block_size, scale);
 	    //cvt_fp32_e5m2_noinf_rne_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	    //cvt_fp32_e5m2_flex_intrinsic(&in[start_index], &out[start_index], block_size, scale);
	  }
	} else {
	  cvt_fp32_e5m2_scalar (&in[start_index], &out[start_index], block_size, scale, rmode);
	}
      }
    } else {
      if ((size % 32) == 0) {
	if (rmode == ROUND_STOCHASTIC) {
	  cvt_fp32_e5m2_stochastic_intrinsic (in, out, size, scale);
	} else {
	  cvt_fp32_e5m2_rne_intrinsic (in, out, size, scale);
	  //cvt_fp32_e5m2_noinf_rne_intrinsic (in, out, size, scale);
        }
      } else {
	int vec_size = ((int) (size / 32)) * 32;

	if (vec_size > 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e5m2_stochastic_intrinsic (in, out, vec_size, scale);
	  } else {
	    cvt_fp32_e5m2_rne_intrinsic (in, out, vec_size, scale);
	    //cvt_fp32_e5m2_noinf_rne_intrinsic (in, out, vec_size, scale);
	    //cvt_fp32_e5m2_flex_intrinsic(in, out, vec_size, scale);
	  }
	}
	cvt_fp32_e5m2_scalar (&in[vec_size], &out[vec_size], size - vec_size, scale, rmode);
      }
    }
  }

  template < typename scalar_t >
    void E5M2_DAZ_Kernel (const scalar_t * __restrict__ in,
			  scalar_t * __restrict__ out,
			  const int size, const scalar_t scale, int rmode) {
    int non_mant_bits = 5 /*exp_bits */  + 1;	/* exponent + sign */
    int lshift = 10 - (8 /*mbits */  - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x00FF;
    unsigned short rne_tie = 0x0180;

    scalar_t scale_reciprocal = 1.0/scale;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      __half_t h;
      float inval = in[gid] * scale;

      h.u = __anyfloat2half_rn (inval);
      /* values above 57344.0, saturate them to +- Infinity */

      unsigned short can_round = ((h.u & 0x7F00) <= 0x7B00) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      /* nearest rounding masks */
      unsigned short rnmask = (h.u & grs_bitmask);
      unsigned short rnmask_tie = (h.u & rne_tie);

      if (is_naninf == 0 && is_normal) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short) rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  h.u += can_round * (rand & 0xFF);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  h.u += can_round * rne_mask *
	    (((rnmask > 0x0080) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  h.u += can_round * rnaz_mask * ((rnmask >= 0x0080) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  h.u += can_round * rntz_mask * ((rnmask > 0x0080) << lshift);
	  /* round to +INF, if rpinf_mask is enabled */
	  h.u += can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0080) << lshift);
	  /* round to -INF, if rminf_mask is enabled */
	  h.u += can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0080) << lshift);
	}
      } else if (is_denorm) {
	/* Flush Denormal */
	h.u = 0;
      }
      /* truncation */
      h.u = (h.u & mask_mant);
      __half2anyfloat (h.u, &out[gid], scale_reciprocal);
    }
  }

  void cvt_fp32_e4m3_rne_intrinsic (const float *__restrict__ in, float *out,
				   int size, float scale) {
    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00); 
    const __m256i vrneadd    = _mm256_set1_epi16 (0x003f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001);
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0080);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x5F00);/* 2^8*1.110 a.k.a 448.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x1800);/* 2^-9, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x2400);/* 2^-6 smallest normal */

    for (int i = 0; i < size; i += 16) {
      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1);
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_);
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_);
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);

      __m256i a_rne_ = _mm256_mask_add_epi16 (a_, maska1_, a_, 
		          _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      /* saturating values beyond +/-MAX */
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska3_, _mm256_or_si256(_mm256_and_si256(a_rne_, vsign), vsatuval));
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska4_, _mm256_and_si256(a_rne_, vsign));
      a_rne_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_rne_, 7), 7);

      bh_ = _mm256_extracti32x4_epi32 (a_rne_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_rne_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e4m3_stochastic_intrinsic (const float *__restrict__ in,
					  float *out, int size, float scale) {
    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00);
    const __m256i vrneadd    = _mm256_set1_epi16 (0x003f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001); 
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0080);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x5F00);/* 2^8*1.110 a.k.a 448.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x1800);/* 2^-9, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x2400);/* 2^-6 smallest normal */

    for (int i = 0; i < size; i += 16) {
      unsigned int rndbuf[16];
      /* generate 128 random bits */
      for (int r = 0; r < 8; r++) {
	rndbuf[r] = (unsigned int) rand_xorshft128plus_scalar (sptr_[r]);
      }
      __m128i rnbits = _mm_load_si128 ((const __m128i *) &rndbuf[0]);

      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1); // might be avx-512
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_); // avx-512
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_); // avx-512
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);


      __m256i a_sr_ = _mm256_mask_add_epi16 (a_, (maska1_ & ~maska5_), a_, 
		           _mm256_srli_epi16 (_mm256_cvtepu8_epi16 (rnbits), 1));
      a_sr_ = _mm256_mask_add_epi16 (a_sr_, maska5_, a_sr_, 
		           _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));

      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska3_, _mm256_or_si256(_mm256_and_si256(a_sr_, vsign), vsatuval));
      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska4_, _mm256_and_si256(a_sr_, vsign));
      a_sr_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_sr_, 7), 7);

      bh_ = _mm256_extracti32x4_epi32 (a_sr_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_sr_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e4m3_scalar (const float *__restrict__ in, float *out,
			    int size, float scale, int rmode) {
    int non_mant_bits = 4 /*exp_bits */  + 1;	/* exponent + sign */
    int lshift = 10 - (8 /*mbits */  - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x007F;
    unsigned short rne_tie = 0x00C0;
    float scale_reciprocal = 1.0 / scale;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      __half_t h;
      float inval = scale * in[gid];

      h.u = __anyfloat2half_rn (inval);
      short exp_h = (short) ((h.u & 0x7C00) >> 10) - 15;
      short sign_h = (h.u & 0x8000);
      short mantissa_h = (h.u & 0x03FF);

      unsigned short can_round = ((h.u & 0x7FFF) < 0x5F00) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      int dshift = 0;

      if (exp_h > 8 || (can_round == 0)) {
	/* Software : saturate values above to +/-448.0 to +/-448.0 */
	mantissa_h = 0x0300;
	exp_h = 8;
	can_round = 0;
      } else if (exp_h < -9) {
	/* flush values below 1-4-3 subnormal range to zero */
	exp_h = -15;
	mantissa_h = 0;
      } else if (exp_h < -6) {
	dshift = (-6 - exp_h);
	/* handle denormals */
	mantissa_h = mantissa_h >> dshift;
        mantissa_h <<= dshift;
      }
      /* nearest rounding masks */
      unsigned short rnmask = (mantissa_h & grs_bitmask);
      unsigned short rnmask_tie = (mantissa_h & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short)
	    rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  mantissa_h += can_round * is_normal * (rand & 0x7F);
	  /* stochastic round:  denormals --> rne rounding */
	  mantissa_h += can_round * is_denorm *
	    (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  mantissa_h += can_round * rne_mask * (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  mantissa_h += can_round * rnaz_mask * ((rnmask >= 0x0040) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  mantissa_h += can_round * rntz_mask * ((rnmask > 0x0040) << lshift);
	  /* round to +INF, if rpinf_mask is enabled */
	  mantissa_h += can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0040) << lshift);
	  /* round to -INF, if rminf_mask is enabled */
	  mantissa_h += can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0040) << lshift);
	}
      }
      /* truncation */
      mantissa_h &= mask_mant;
      mantissa_h += ((exp_h + 15) << 10);
      mantissa_h |= sign_h;
      h.u = mantissa_h;
      float f;

      __half2anyfloat (h.u, &f);
      out[gid] = (f * scale_reciprocal);
    }
  }

  template < typename scalar_t >
    void E4M3_Kernel (const scalar_t * __restrict__ in,
	  	      scalar_t * __restrict__ out,
		      const int size,
		      const scalar_t in_scale,
		      bool block_norm,
		      int block_size, int rmode) {
    float scale = in_scale;
    float fmax = std::numeric_limits<scalar_t>::max();
    if (scale > fmax) {
      fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
      exit(1);
    }

    if (block_norm == true) {
      int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
      for (int b = 0; b < nblocks; b++) {
	int start_index = (b * block_size);

	/* handle the last block */
	if (start_index + block_size > size)
	  block_size = (size - start_index);

	float maxval = 0.0;

#pragma omp parallel for reduction (max:maxval)
	for (int gid = start_index; gid < start_index + block_size; gid++) {
	  maxval = (maxval < fabs (in[gid])) ? fabs (in[gid]) : maxval;
	}
	__float_t f;

	f.f = maxval;
	f.u = (f.u & 0x7F800000);
	scale = 2.0 * f.f;
	scale /= 8.0;

	if ((block_size % 32) == 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e4m3_stochastic_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  } else {
	    cvt_fp32_e4m3_rne_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  }
	} else {
	  cvt_fp32_e4m3_scalar (&in[start_index], &out[start_index], block_size, scale, rmode);
	}
      }
    } else {
      if ((size % 32) == 0) {
	if (rmode == ROUND_STOCHASTIC) {
	  cvt_fp32_e4m3_stochastic_intrinsic (in, out, size, scale);
	} else {
	  cvt_fp32_e4m3_rne_intrinsic (in, out, size, scale);
	}
      } else {
	int vec_size = ((int) (size / 32)) * 32;

	if (vec_size > 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e4m3_stochastic_intrinsic (in, out, vec_size, scale);
	  } else {
	    cvt_fp32_e4m3_rne_intrinsic (in, out, vec_size, scale);
	  }
	}
	cvt_fp32_e4m3_scalar (&in[vec_size], &out[vec_size], size - vec_size, scale, rmode);
      }
    }
  }

  void cvt_fp32_e4m3_ieee_rne_intrinsic (const float *__restrict__ in, float *out,
				   int size, float scale) {
    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00); 
    const __m256i vrneadd    = _mm256_set1_epi16 (0x003f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001);
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0080);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x5B80);/* 2^7*1.111 a.k.a 240.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x1800);/* 2^-9, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x2400);/* 2^-6 smallest normal */

    for (int i = 0; i < size; i += 16) {
      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1);
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_);
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_);
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);

      __m256i a_rne_ = _mm256_mask_add_epi16 (a_, maska1_, a_, 
		          _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      /* saturating values beyond +/-MAX */
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska3_, _mm256_or_si256(_mm256_and_si256(a_rne_, vsign), vsatuval));
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska4_, _mm256_and_si256(a_rne_, vsign));

      a_rne_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_rne_, 7), 7);

      bh_ = _mm256_extracti32x4_epi32 (a_rne_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_rne_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e4m3_ieee_stochastic_intrinsic (const float *__restrict__ in,
					  float *out, int size, float scale) {
    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00);
    const __m256i vrneadd    = _mm256_set1_epi16 (0x003f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001); 
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0080);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x5B80);/* 2^7*1.111 a.k.a 240.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x1800);/* 2^-9, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x2400);/* 2^-6 smallest normal */

    for (int i = 0; i < size; i += 16) {
      unsigned int rndbuf[16];
      /* generate 128 random bits */
      for (int r = 0; r < 8; r++) {
	rndbuf[r] = (unsigned int) rand_xorshft128plus_scalar (sptr_[r]);
      }
      __m128i rnbits = _mm_load_si128 ((const __m128i *) &rndbuf[0]);

      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1);
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_);
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_);
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);


      __m256i a_sr_ = _mm256_mask_add_epi16 (a_, (maska1_ & ~maska5_), a_, 
		           _mm256_srli_epi16 (_mm256_cvtepu8_epi16 (rnbits), 1));
      a_sr_ = _mm256_mask_add_epi16 (a_sr_, maska5_, a_sr_, 
		           _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska3_, _mm256_or_si256(_mm256_and_si256(a_sr_, vsign), vsatuval));
      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska4_, _mm256_and_si256(a_sr_, vsign));

      a_sr_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_sr_, 7), 7);

      bh_ = _mm256_extracti32x4_epi32 (a_sr_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_sr_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e4m3_ieee_scalar (const float *__restrict__ in, float *out,
			    int size, float scale, int rmode) {
    int non_mant_bits = 4 /*exp_bits */  + 1;	/* exponent + sign */
    int lshift = 10 - (8 /*mbits */  - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x007F;
    unsigned short rne_tie = 0x00C0;
    float scale_reciprocal = 1.0 / scale;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      __half_t h;
      float inval = scale * in[gid];

      h.u = __anyfloat2half_rn (inval);
      short exp_h = (short) ((h.u & 0x7C00) >> 10) - 15;
      short sign_h = (h.u & 0x8000);
      short mantissa_h = (h.u & 0x03FF);

      unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      int dshift = 0;

      if (exp_h > 7 || (can_round == 0)) {
	mantissa_h = 0x380;
	exp_h = 7; 
	can_round = 0;
      } else if (exp_h < -9) {
	/* flush values below 1-4-3 subnormal range to zero */
	exp_h = -15;
	mantissa_h = 0;
      } else if (exp_h < -6) {
	dshift = (-6 - exp_h);
	/* handle denormals */
	mantissa_h = mantissa_h >> dshift;
	mantissa_h <<= dshift;
      }
      /* nearest rounding masks */
      unsigned short rnmask = (mantissa_h & grs_bitmask);
      unsigned short rnmask_tie = (mantissa_h & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short)
	    rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  mantissa_h += can_round * is_normal * (rand & 0x7F);
	  /* stochastic round:  denormals --> rne rounding */
	  mantissa_h += can_round * is_denorm *
	    (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  mantissa_h += can_round * rne_mask *
	    (((rnmask > 0x0040) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  mantissa_h += can_round * rnaz_mask * ((rnmask >= 0x0040) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  mantissa_h += can_round * rntz_mask * ((rnmask > 0x0040) << lshift);
	  /* round to +INF, if rpinf_mask is enabled */
	  mantissa_h += can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0040) << lshift);
	  /* round to -INF, if rminf_mask is enabled */
	  mantissa_h += can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0040) << lshift);
	}
      }
      /* truncation */
      mantissa_h &= mask_mant;
      mantissa_h += ((exp_h + 15) << 10);
      mantissa_h |= sign_h;
      h.u = mantissa_h;
      float f;

      __half2anyfloat (h.u, &f);
      out[gid] = (f * scale_reciprocal);
    }
  }

  template < typename scalar_t >
    void E4M3_IEEE_Kernel (const scalar_t * __restrict__ in,
	  	      scalar_t * __restrict__ out,
		      const int size,
		      const scalar_t in_scale,
		      bool block_norm,
		      int block_size, int rmode) {
    float scale = in_scale;
    float fmax = std::numeric_limits<scalar_t>::max();
    if (scale > fmax) {
      fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
      exit(1);
    }

    if (block_norm == true) {
      int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
      for (int b = 0; b < nblocks; b++) {
	int start_index = (b * block_size);

	/* handle the last block */
	if (start_index + block_size > size)
	  block_size = (size - start_index);

	float maxval = 0.0;

#pragma omp parallel for reduction (max:maxval)
	for (int gid = start_index; gid < start_index + block_size; gid++) {
	  maxval = (maxval < fabs (in[gid])) ? fabs (in[gid]) : maxval;
	}
	__float_t f;

	f.f = maxval;
	f.u = (f.u & 0x7F800000);
	scale = 2.0 * f.f;
	scale /= 8.0;

	if ((block_size % 32) == 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e4m3_ieee_stochastic_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  } else {
	    cvt_fp32_e4m3_ieee_rne_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  }
	} else {
	  cvt_fp32_e4m3_ieee_scalar (&in[start_index], &out[start_index], block_size, scale, rmode);
	}
      }
    } else {
      if ((size % 32) == 0) {
	if (rmode == ROUND_STOCHASTIC) {
	  cvt_fp32_e4m3_ieee_stochastic_intrinsic (in, out, size, scale);
	} else {
	  cvt_fp32_e4m3_ieee_rne_intrinsic (in, out, size, scale);
	}
      } else {
	int vec_size = ((int) (size / 32)) * 32;
	if (vec_size > 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e4m3_ieee_stochastic_intrinsic (in, out, vec_size, scale);
	  } else {
	    cvt_fp32_e4m3_ieee_rne_intrinsic (in, out, vec_size, scale);
	  }
	}
	cvt_fp32_e4m3_ieee_scalar (&in[vec_size], &out[vec_size], size - vec_size, scale, rmode);
      }
    }
  }

  void cvt_fp32_e3m4_rne_intrinsic (const float *__restrict__ in,
				       float *out, int size, float scale) {

    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00); 
    const __m256i vrneadd    = _mm256_set1_epi16 (0x001f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001);
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0040);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x4F80);/* 2^4*1.1110 a.k.a 30.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x2400);/* 2^-6, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x3400);/* 2^-2 smallest normal */

    for (int i = 0; i < size; i += 16) {
      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1);
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_);
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_);
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);

      __m256i a_rne_ = _mm256_mask_add_epi16 (a_, maska1_, a_, 
		          _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));
      /* saturating values beyond +/-MAX */
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska3_, _mm256_or_si256(_mm256_and_si256(a_rne_, vsign), vsatuval));
      a_rne_ = _mm256_mask_mov_epi16 (a_rne_, maska4_, _mm256_and_si256(a_rne_, vsign));

      a_rne_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_rne_, 6), 6);

      bh_ = _mm256_extracti32x4_epi32 (a_rne_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_rne_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e3m4_stochastic_intrinsic (const float *__restrict__ in,
					      float *out, int size,
					      float scale) {

    const __m256i vnaninf    = _mm256_set1_epi16 (0x7c00);
    const __m256i vrneadd    = _mm256_set1_epi16 (0x001f);
    const __m256i vfixup     = _mm256_set1_epi16 (0x0001); 
    const __m256i vfixupmask = _mm256_set1_epi16 (0x0040);
    const __m256i vzero      = _mm256_set1_epi16 (0x0000);
    const __m256i vsign      = _mm256_set1_epi16 (0x8000);
    const __m256i vsatuval   = _mm256_set1_epi16 (0x4F80);/* 2^4*1.1110 a.k.a 30.0, largest value */
    const __m256i vflush     = _mm256_set1_epi16 (0x2400);/* 2^-6, smallest denormal */
    const __m256i vxdnorm    = _mm256_set1_epi16 (0x3400);/* 2^-2 smallest normal */

    for (int i = 0; i < size; i += 16) {
      unsigned int rndbuf[16];
      /* generate 128 random bits */
      for (int r = 0; r < 8; r++) {
	rndbuf[r] = (unsigned int) rand_xorshft128plus_scalar (sptr_[r]);
      }
      __m128i rnbits = _mm_load_si128 ((const __m128i *) &rndbuf[0]);

      __m256 s_  = _mm256_set1_ps (scale);
      __m256 sr_ = _mm256_set1_ps (1.0 / scale);
      __m256 b   = _mm256_loadu_ps (&in[i]);
      __m256 a   = _mm256_loadu_ps (&in[i + 8]);

      b = _mm256_mul_ps (b, s_);
      a = _mm256_mul_ps (a, s_);

      __m128i ah_ = _mm256_cvtps_ph (a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m128i bh_ = _mm256_cvtps_ph (b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256i a_  = _mm256_inserti32x4 (_mm256_inserti32x4 (_mm256_setzero_si256 (), bh_, 0), ah_, 1);
      const __mmask16 maska1_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vnaninf, _MM_CMPINT_NE);
      const __mmask16 maska2_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
      const __mmask16 maska3_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, _mm256_set1_epi16(0x7FFF)), vsatuval, _MM_CMPINT_NLT);
      const __mmask16 maska4_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vflush, _MM_CMPINT_LT);
      const __mmask16 maska5_ =	_mm256_cmp_epi16_mask (_mm256_and_si256 (a_, vnaninf), vxdnorm, _MM_CMPINT_LT);
      /* Handle denormals, shift the mantissa before rounding */
      __m256i v_shft_ =	_mm256_sub_epi16 (
		           _mm256_srli_epi16 (vxdnorm, 10), 
			   _mm256_srli_epi16 (_mm256_and_si256 (a_, vnaninf), 10));
      v_shft_ = _mm256_mask_mov_epi16 (vzero, maska5_, v_shft_);
      a_ = _mm256_mask_srlv_epi16 (a_, maska5_, a_, v_shft_);
      a_ = _mm256_mask_sllv_epi16 (a_, maska5_, a_, v_shft_);

      __m256i a_sr_ = _mm256_mask_add_epi16 (a_, (maska1_ & ~maska5_), a_, 
		           _mm256_srli_epi16 (_mm256_cvtepu8_epi16 (rnbits), 1));
      a_sr_ = _mm256_mask_add_epi16 (a_sr_, maska5_, a_sr_, 
		           _mm256_mask_add_epi16 (vrneadd, maska2_, vrneadd, vfixup));

      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska3_, _mm256_or_si256(_mm256_and_si256(a_sr_, vsign), vsatuval));
      a_sr_ = _mm256_mask_mov_epi16 (a_sr_, maska4_, _mm256_and_si256(a_sr_, vsign));
      a_sr_ = _mm256_slli_epi16 (_mm256_srli_epi16 (a_sr_, 6), 6);

      bh_ = _mm256_extracti32x4_epi32 (a_sr_, 0);
      ah_ = _mm256_extracti32x4_epi32 (a_sr_, 1);
      b = _mm256_cvtph_ps (bh_);
      a = _mm256_cvtph_ps (ah_);
      _mm256_storeu_ps (&out[i], _mm256_mul_ps (b, sr_));
      _mm256_storeu_ps (&out[i + 8], _mm256_mul_ps (a, sr_));
    }
  }

  void cvt_fp32_e3m4_scalar (const float *__restrict__ in, float *out,
				int size, float scale, int rmode) {
    int non_mant_bits = 3 /*exp_bits */  + 1;	/* exponent + sign */
    int lshift = 10 - (8 /*mbits */  - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x003F;
    unsigned short rne_tie = 0x0060;
    float scale_reciprocal = 1.0 / scale;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      __half_t h;
      float inval = scale * in[gid];

      h.u = __anyfloat2half_rn (inval);
      short exp_h = (short) ((h.u & 0x7C00) >> 10) - 15;
      short sign_h = (h.u & 0x8000);
      short mantissa_h = (h.u & 0x03FF);

      unsigned short can_round = ((h.u & 0x7FFF) < 0x4F80) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      int dshift = 0;
      if (exp_h > 4 || (can_round == 0)) {
	/* Software : saturate values above +/-30.0 to +/-30.0 */
	mantissa_h = 0x0380;
	exp_h = 4;
	can_round = 0;
      } else if (exp_h < -6) {
	exp_h = -15;
	mantissa_h = 0;
      } else if (exp_h < -2) {
	dshift = (-2 - exp_h);
	/* handle denormals */
	mantissa_h = mantissa_h >> dshift;
	mantissa_h <<= dshift;
      }
      /* nearest rounding masks */
      unsigned short rnmask = (mantissa_h & grs_bitmask);
      unsigned short rnmask_tie = (mantissa_h & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short)
	    rand_xorshft128plus_scalar (sptr_[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  mantissa_h += can_round * is_normal * (rand & 0x3F);
	  /* stochastic round:  denormals --> rne rounding */
	  mantissa_h += can_round * is_denorm *
	                (((rnmask > 0x0020) || (rnmask_tie == rne_tie)) << lshift);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  mantissa_h += can_round * rne_mask *
	                (((rnmask > 0x0020) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  mantissa_h += can_round * rnaz_mask * ((rnmask >= 0x0020) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  mantissa_h += can_round * rntz_mask * ((rnmask > 0x0020) << lshift);
	  /* round to +INF, if rpinf_mask is enabled */
	  mantissa_h += can_round * rpinf_mask * (h.f > 0) * ((rnmask >= 0x0020) << lshift);
	  /* round to -INF, if rminf_mask is enabled */
	  mantissa_h += can_round * rminf_mask * (h.f < 0) * ((rnmask >= 0x0020) << lshift);
	}
      }
      /* truncation */
      mantissa_h &= mask_mant;
      mantissa_h += ((exp_h + 15) << 10);
      mantissa_h |= sign_h;
      h.u = mantissa_h;
      float f;

      __half2anyfloat (h.u, &f);
      out[gid] = (f * scale_reciprocal);
    }
  }

  template < typename scalar_t >
    void E3M4_Kernel (const scalar_t * __restrict__ in,
	  	      scalar_t * __restrict__ out,
		      const int size,
		      const scalar_t in_scale,
		      bool block_norm,
		      int block_size, int rmode) {
    float scale = in_scale;
    float fmax = std::numeric_limits<scalar_t>::max();
    if (scale > fmax) {
      fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
      exit(1);
    }

    if (block_norm == true) {
      int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
      for (int b = 0; b < nblocks; b++) {
	int start_index = (b * block_size);

	/* handle the last block */
	if (start_index + block_size > size)
	  block_size = (size - start_index);

	float maxval = 0.0;

#pragma omp parallel for reduction (max:maxval)
	for (int gid = start_index; gid < start_index + block_size; gid++) {
	  maxval = (maxval < fabs (in[gid])) ? fabs (in[gid]) : maxval;
	}
	__float_t f;

	f.f = maxval;
	f.u = (f.u & 0x7F800000);
	scale = 2.0 * f.f;

	if ((block_size % 32) == 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e3m4_stochastic_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  } else {
	    cvt_fp32_e3m4_rne_intrinsic (&in[start_index], &out[start_index], block_size, scale);
	  }
	} else {
	  cvt_fp32_e3m4_scalar (&in[start_index], &out[start_index], block_size, scale, rmode);
	}
      }
    } else {
      if ((size % 32) == 0) {
	if (rmode == ROUND_STOCHASTIC) {
	  cvt_fp32_e3m4_stochastic_intrinsic (in, out, size, scale);
	} else {
	  cvt_fp32_e3m4_rne_intrinsic (in, out, size, scale);
	}
      } else {
	int vec_size = ((int) (size / 32)) * 32;
	if (vec_size > 0) {
	  if (rmode == ROUND_STOCHASTIC) {
	    cvt_fp32_e3m4_stochastic_intrinsic (in, out, vec_size, scale);
	  } else {
	    cvt_fp32_e3m4_rne_intrinsic (in, out, vec_size, scale);
	  }
	}
	cvt_fp32_e3m4_scalar (&in[vec_size], &out[vec_size],
			      size - vec_size, scale, rmode);
      }
    }
  }

  void cvt_fp32_fp4_nearest_scalar (const float *__restrict__ in, float *out,
				int size, float scale) {

    float scale_reciprocal = 1.0 / scale;

#pragma omp parallel for
    for (int gid = 0; gid < size; gid++) {
      __float_t f;
      float inval = scale * in[gid];

      f.f = inval;
      int exp_f = (int)(((f.u & 0x7F800000) >> 23) - 127);
      int sign_f = (f.u & 0x80000000);
      /* see if round up works! */
      if (exp_f < 0 && (exp_f%2)) f.f *= 1.6;
      /* saturate */
      if (exp_f > 0) f.u = (sign_f | (127 << 23));
      f.u &= 0xFF800000;
      /* extract the new exponent */
      exp_f = (int)(((f.u & 0x7F800000) >> 23) - 127);
      /* round up did not work, round down */
      if (exp_f < 0 && (exp_f%2)) f.u = (sign_f | ((exp_f + 126) << 23));
      /* flush values smaller than 2^-12 to zero */
      //if (exp_f < -6) f.u = 0;
      if (exp_f < -12) f.u = 0;
      out[gid] = (f.f * scale_reciprocal);
    }
  }

  template < typename scalar_t >
    void FP4_Nearest_Kernel (const scalar_t * __restrict__ in,
	  	      scalar_t * __restrict__ out,
		      const int size,
		      const scalar_t in_scale,
		      bool block_norm,
		      int block_size) {
    float scale = in_scale;
    float fmax = std::numeric_limits<scalar_t>::max();
    if (scale > fmax) {
      fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
      exit(1);
    }

    if (block_norm == true) {
      int nblocks = (size + (block_size - 1)) / block_size;

#pragma omp parallel for
      for (int b = 0; b < nblocks; b++) {
	int start_index = (b * block_size);

	/* handle the last block */
	if (start_index + block_size > size)
	  block_size = (size - start_index);

	float maxval = 0.0;

#pragma omp parallel for reduction (max:maxval)
	for (int gid = start_index; gid < start_index + block_size; gid++) {
	  maxval = (maxval < fabs (in[gid])) ? fabs (in[gid]) : maxval;
	}
	/* FP4 max value is 1.0 */
	scale = 1.0/maxval;
	cvt_fp32_fp4_nearest_scalar(&in[start_index], &out[start_index], block_size, scale);
      }
    } else {
      cvt_fp32_fp4_nearest_scalar (&in[0], &out[0], size, scale);
    }
  }

  std::vector < torch::Tensor > fpemu_common_function (torch::Tensor input,
					std::string mode,
					int size,
					bool inplace,
					float scale,
					bool block_norm,
					int block_size) {

    torch::Tensor output;
    if (!inplace)
      output = torch::zeros_like (input);

    if (!mode.compare ("E5M2_RTZ")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_RTZ);
    } else if (!mode.compare ("E5M2_RNE")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_RNE);
    } else if (!mode.compare ("E5M2_STOCHASTIC")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_STOCHASTIC);
    } else if (!mode.compare ("E5M2_RNAZ")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_RNAZ);
    } else if (!mode.compare ("E5M2_RNTZ")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
		            size, scale, block_norm, block_size, ROUND_RNTZ);
    } else if (!mode.compare ("E5M2_RPINF")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
		            (inplace) ? input.data_ptr <
			    float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_PINF);
    } else if (!mode.compare ("E5M2_RNINF")) {
      E5M2_Kernel < float >(input.data_ptr < float >(),
		            (inplace) ? input.data_ptr <
		            float >() : output.data_ptr < float >(),
			    size, scale, block_norm, block_size, ROUND_NINF);
    } else if (!mode.compare ("E5M2_DAZ_RNE")) {
      E5M2_DAZ_Kernel < float >(input.data_ptr < float >(),
				(inplace) ? input.data_ptr <
				float >() : output.data_ptr <
				float >(), size, scale, ROUND_RNE);
    } else if (!mode.compare ("E5M2_DAZ_STOCHASTIC")) {
      E5M2_DAZ_Kernel < float >(input.data_ptr < float >(),
				(inplace) ? input.data_ptr <
				float >() : output.data_ptr <
				float >(), size, scale, ROUND_STOCHASTIC);
    } else if (!mode.compare ("E5M2_DAZ_RNAZ")) {
      E5M2_DAZ_Kernel < float >(input.data_ptr < float >(),
				(inplace) ? input.data_ptr <
				float >() : output.data_ptr <
				float >(), size, scale, ROUND_RNAZ);
    } else if (!mode.compare ("E5M2_DAZ_RNTZ")) {
      E5M2_DAZ_Kernel < float >(input.data_ptr < float >(),
				(inplace) ? input.data_ptr <
				float >() : output.data_ptr <
				float >(), size, scale, ROUND_RNTZ);
    } else if (!mode.compare ("FLOAT16_RNE")) {
      FLOAT16_Kernel < float >(input.data_ptr < float >(),
			      (inplace) ? input.data_ptr <
			      float >() : output.data_ptr < float >(),
			      size, ROUND_RNE, 0);
    } else if (!mode.compare ("FLOAT16_STOCHASTIC")) {
      FLOAT16_Kernel < float >(input.data_ptr < float >(),
			      (inplace) ? input.data_ptr <
			      float >() : output.data_ptr < float >(),
			      size, ROUND_STOCHASTIC, 0);
    } else if (!mode.compare ("FLOAT16_DAZ_RNE")) {
      FLOAT16_Kernel < float >(input.data_ptr < float >(),
			      (inplace) ? input.data_ptr <
			      float >() : output.data_ptr < float >(),
			      size, ROUND_RNE, 1);
    } else if (!mode.compare ("BFLOAT16_RNE")) {
      BFLOAT16_Kernel < float >(input.data_ptr < float >(),
			       (inplace) ? input.data_ptr <
			       float >() : output.data_ptr <
			       float >(), size, ROUND_RNE);
    } else if (!mode.compare ("BFLOAT16_STOCHASTIC")) {
      BFLOAT16_Kernel < float >(input.data_ptr < float >(),
			       (inplace) ? input.data_ptr <
			       float >() : output.data_ptr <
			       float >(), size, ROUND_STOCHASTIC);
    } else if (!mode.compare ("E4M3_IEEE_RNE")) {
      E4M3_IEEE_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_RNE);
    } else if (!mode.compare ("E4M3_IEEE_STOCHASTIC")) {
      E4M3_IEEE_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_STOCHASTIC);
    } else if (!mode.compare ("E4M3_RNE")) {
      E4M3_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_RNE);
    } else if (!mode.compare ("E4M3_STOCHASTIC")) {
      E4M3_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_STOCHASTIC);
    } else if (!mode.compare ("E3M4_RNE")) {
      E3M4_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_RNE);
    } else if (!mode.compare ("E3M4_STOCHASTIC")) {
      E3M4_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size, ROUND_STOCHASTIC);
    } else if (!mode.compare ("FP4_NEAREST")) {
      FP4_Nearest_Kernel < float >(input.data_ptr < float >(),
			    (inplace) ? input.data_ptr <
			    float >() : output.data_ptr <
			    float >(), size, scale, block_norm,
			    block_size);
    }
    
    if (!inplace) {
      return {
      output,};
    } else {
      return {
      input,};
    }
  }

}//namespace

std::vector < torch::Tensor > fpemu_forward (torch::Tensor input,
					std::string mode,
					int size,
					bool inplace,
					float scale,
					bool block_norm,
					int block_size) {
  if (block_norm == true && block_size != size) {
    if (size % block_size) {
      block_norm = false;
      block_size = 1;
    }
  }
  return fpemu_common_function (input, mode, size, inplace, scale, block_norm,
				   block_size);
}

std::vector < torch::Tensor > fpemu_backward (torch::Tensor grad,
					std::string mode,
					int size,
					bool inplace,
					float scale,
					bool block_norm,
					int block_size) {
  if (block_norm == true && block_size != size) {
    if (size % block_size) {
      block_norm = false;
      block_size = 1;
    }
  }
  return fpemu_common_function (grad, mode, size, inplace, scale, block_norm,
				   block_size);
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, m) {
  m.def ("forward", &fpemu_forward, "FPEmu forward");
  m.def ("backward", &fpemu_backward, "FPEmu backward");
}