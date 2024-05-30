/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <torch/extension.h>
#include <ATen/cuda/Exceptions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CUBLOCK_SIZE 256
enum ROUNDING_MODES { ROUND_RTZ = 0, ROUND_RNE = 1, ROUND_STOCHASTIC =
    2, ROUND_RNAZ = 3, ROUND_RNTZ = 4, ROUND_PINF = 5, ROUND_NINF = 6 };

namespace {

  typedef union half_t {
    unsigned short u;
      at::Half f;
  } __half_t;

  typedef union ufloat32 {
    unsigned u;
    float f;
  } __float_t;

/* this implementation of xoroshiro128++ PRNG is borrowed from here: 
    http://prng.di.unimi.it/xoshiro128plusplus.c 
    main page: http://prng.di.unimi.it/ 
*/
  __device__ static uint32_t s1[4] =
    { 1387366120, 279844183, 888998500, 1099633400 };
  __device__ static uint32_t s2[4] =
    { 2034269327, 2125325156, 1209715489, 193165672 };
  __device__ static uint32_t s3[4] =
    { 1555452618, 650181557, 883695203, 62767784 };
  __device__ static uint32_t s4[4] =
    { 419524804, 2146478152, 480059239, 1468956197 };
  __device__ static uint32_t s5[4] =
    { 1252084877, 500390994, 977516591, 1950666000 };
  __device__ static uint32_t s6[4] =
    { 393659750, 834151069, 1477014702, 734008143 };
  __device__ static uint32_t s7[4] =
    { 1983400973, 116410309, 2110188261, 2019272068 };
  __device__ static uint32_t s8[4] =
    { 187709636, 28336299, 419632041, 1774181187 };
  __device__ static uint32_t s9[4] =
    { 702309618, 407781555, 1512057936, 1868769368 };
  __device__ static uint32_t s10[4] =
    { 510001215, 966559856, 776583255, 147562106 };
  __device__ static uint32_t s11[4] =
    { 127180605, 1881312534, 478635452, 814821902 };
  __device__ static uint32_t s12[4] =
    { 733990058, 1889991804, 1108257970, 1093480892 };
  __device__ static uint32_t s13[4] =
    { 427374380, 416747337, 558000409, 1594848927 };
  __device__ static uint32_t s14[4] =
    { 444870959, 1595722866, 1064124488, 363710254 };
  __device__ static uint32_t s15[4] =
    { 703721499, 389640783, 1002360059, 1427395742 };
  __device__ static uint32_t s16[4] =
    { 1295231497, 1254972431, 1423497865, 861918264 };

  __device__ static uint32_t *sptr[16] =
    { s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16 };

  __device__ __forceinline__ uint32_t rotl_ (const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
  }

  __device__ __forceinline__ uint32_t _rand_xorshft128plus_with_seed (uint32_t
								      * ps) {
    const uint32_t result_plus = ps[0] + ps[3];
    const uint32_t t = ps[1] << 9;

    ps[2] ^= ps[0];
    ps[3] ^= ps[1];
    ps[1] ^= ps[2];
    ps[0] ^= ps[3];

    ps[2] ^= t;

    ps[3] = rotl_ (ps[3], 11);

    return result_plus;
  }

  template < typename scalar_t >
    __device__ __forceinline__ float __anyfloat2float_rn (scalar_t a_) {
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
    __device__ __forceinline__ void __float2anyfloat_rn (float f_,
							 scalar_t * out) {
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
    __device__ __forceinline__ at::Half __anyfloat2half_rn (scalar_t f_) {
    at::Half h_;
    if (std::is_same < scalar_t, double >::value) {
      h_ = __float2half_rn (__double2float_rn (f_));
    } else if (std::is_same < scalar_t, float >::value) {
      h_ = __float2half_rn (f_);
    } else if (std::is_same < scalar_t, at::Half >::value) {
      h_ = (at::Half) f_;
    }
    return h_;
  }

  template < typename scalar_t >
    __device__ __forceinline__ void __half2anyfloat (at::Half h_,
						     scalar_t * out) {
    scalar_t f_;

    if (std::is_same < scalar_t, double >::value) {
      f_ = (scalar_t) __half2float ((at::Half) h_);
    } else if (std::is_same < scalar_t, float >::value) {
      f_ = __half2float (h_);
    } else if (std::is_same < scalar_t, at::Half >::value) {
      f_ = (at::Half) h_;
    }
    *out = f_;
  }

  template < typename scalar_t >
    __device__ void absmax_block (const scalar_t * in,
				  float *sdata, const int size) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0.0f;

    if (i < size) {
      sdata[tid] =
	fmaxf (fabsf (sdata[tid]), fabsf (__anyfloat2float_rn (in[i])));
    }
    __syncthreads ();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0) {
	sdata[tid] = fmaxf (fabsf (sdata[tid]), fabsf (sdata[tid + s]));
      }
      __syncthreads ();
    }
  }

  __device__ static inline float atomicMaxf (float *address, float val) {
    int *address_as_int = (int *) address;
    int old = *address_as_int, assumed;

    while (val > __int_as_float (old)) {
      assumed = old;
      old = atomicCAS (address_as_int, assumed, __float_as_int (val));
    }
    return __int_as_float (old);
  }

  template < typename scalar_t >
    __global__ void absmax0 (scalar_t * g_data, float *g_odata,
			     unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0.0f;

    if (i < n) {
      sdata[tid] =
	fmaxf (fabsf (sdata[tid]), fabsf (__anyfloat2float_rn (g_data[i])));
    }
    __syncthreads ();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0) {
	sdata[tid] = fmaxf (fabsf (sdata[tid]), fabsf (sdata[tid + s]));
      }
      __syncthreads ();
    }
    __syncthreads ();
    if (tid == 0) {
      atomicMaxf (&g_odata[0], sdata[0]);
    }
  }

  template < typename scalar_t >
    __global__ void reduce0 (scalar_t * g_data, float *g_odata,
			     unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
      sdata[tid] = __anyfloat2float_rn (g_data[i]);
      sdata[tid + blockDim.x] =
	__anyfloat2float_rn (g_data[i]) * __anyfloat2float_rn (g_data[i]);
    } else {
      sdata[tid] = 0.0f;
      sdata[tid + blockDim.x] = 0.0f;
    }
    __syncthreads ();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if ((tid % (2 * s)) == 0) {
	sdata[tid] += sdata[tid + s];
	sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
      }
      __syncthreads ();
    }
    __syncthreads ();
    if (tid == 0) {
      atomicAdd (&g_odata[0], sdata[0]);
      atomicAdd (&g_odata[1], sdata[blockDim.x]);
    }
  }


  template < typename scalar_t >
    __global__ void BFLOAT16_Kernel (const scalar_t * in,
			scalar_t * out,
			const int size, int rmode) {
    int lshift = 16;
    unsigned int mask_mant = (unsigned int) (0xFFFFFFFF << lshift);
    unsigned int grs_bitmask = 0x0000FFFF;
    unsigned int rne_tie = 0x00018000;

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */

    if (rmode == ROUND_RNE)
      rne_mask = 1;
    if (rmode == ROUND_RNAZ)
      rnaz_mask = 1;
    if (rmode == ROUND_RNTZ)
      rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC)
      sr_mask = 1;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __float_t uf;

      uf.f = __anyfloat2float_rn (in[gid]);
      unsigned int is_normal = (((uf.u & 0x7F800000) <= 0x7F000000)
				&& ((uf.u & 0x7F800000) >= 0x00800000)) ? 1 : 0;
      unsigned int is_denorm = ((uf.u & 0x7F800000) == 0x0) ? 1 : 0;
      unsigned int is_naninf = ((uf.u & 0x7F800000) == 0x7F800000) ? 1 : 0;

      /* nearest rounding masks */
      unsigned int rnmask = (uf.u & grs_bitmask);
      unsigned int rnmask_tie = (uf.u & rne_tie);

      if (is_naninf == 0 && is_normal) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned int rand = _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
	  /* apply stochastic rounding before truncation if sr_mask is enabled */
	  uf.u += (rand & 0x0000FFFF);
	} else {
	  /* round to nearest even, if rne_mask is enabled */
	  uf.u += rne_mask *
	    (((rnmask > 0x00008000) || (rnmask_tie == rne_tie)) << lshift);
	  /* round to nearest away from zero, if rnaz_mask is enabled */
	  uf.u += rnaz_mask * ((rnmask >= 0x00008000) << lshift);
	  /* round to nearest towards zero, if rntz_mask is enabled */
	  uf.u += rntz_mask * ((rnmask > 0x00008000) << lshift);
	}
      } else if (is_denorm) {
	/* Flush Denormal */
	uf.u = 0;
      }
      /* truncation */
      uf.u = (uf.u & mask_mant);

      __float2anyfloat_rn (uf.f, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void FLOAT16_Kernel (const scalar_t * in,
			scalar_t * out,
			const int size,
			int rmode, int no_denorm) {
    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      if (std::is_same < scalar_t, at::Half >::value || rne_mask) {
	__half_t h;

	at::Half hval;
	hval = __anyfloat2half_rn (in[gid]);
	h.f = hval;
	unsigned short not_denorm = ((((h.u & 0x7FFF) >> 10) & 0x1F) > 0);
	unsigned short is_denorm = (not_denorm == 0) ? 1 : 0;

	h.u *= !(is_denorm * no_denorm);
	__half2anyfloat (h.f, &out[gid]);
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
	  unsigned int rand = (unsigned int)
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
	  if (exp_h < -14) {
	    /* handle denormal values */
	    h.f = __anyfloat2half_rn (in[gid]);
	  } else {
	    exp_h += 15;
	    mantissa_h |= (exp_h << 23);
	    mantissa_h |= (sign_h >> 3);
	    mantissa_h += (rand & 0x00001FFF);
	    h.u = (unsigned short) (mantissa_h >> 13);
	  }
	}
	__half2anyfloat (h.f, &out[gid]);
      }
    }
  }

  template < typename scalar_t >
    __global__ void E5M2_Kernel (const scalar_t * __restrict__ in,
			scalar_t * __restrict__ out,
			const int size,
			const scalar_t in_scale,
			bool block_norm,
			int mbits,
			int exp_bits, int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

    unsigned short rne_mask = 0;	/* round to nearest even mask */
    unsigned short rnaz_mask = 0;	/* round to nearest away from zero mask */
    unsigned short rntz_mask = 0;	/* round to nearest towards zero mask */
    unsigned short rpinf_mask = 0;	/* round to +INF */
    unsigned short rminf_mask = 0;	/* round to -INF */
    unsigned short sr_mask = 0;	/* stochastic rounding mask */

    if (rmode == ROUND_RNE) rne_mask = 1;
    if (rmode == ROUND_RNAZ) rnaz_mask = 1;
    if (rmode == ROUND_RNTZ) rntz_mask = 1;
    if (rmode == ROUND_STOCHASTIC) sr_mask = 1;
    if (rmode == ROUND_PINF) rpinf_mask = 1;
    if (rmode == ROUND_NINF) rminf_mask = 1;

    unsigned short mask_mant = (unsigned short) (0xFFFF << lshift);
    unsigned short grs_bitmask = 0x00FF;
    unsigned short rne_tie = 0x0180;

    extern __shared__ float sdata[];
    scalar_t scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      __float_t f;
      f.f = (float) sdata[0];
      f.u = (f.u & 0x7F800000);
      scale = 2.0 * f.f;
      scale /= 16384.0; 
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = in[gid] * scale;
      __half hval = __anyfloat2half_rn (inval);
      h.f = hval;

      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;
      unsigned short can_round = ((h.u & 0x7F00) <= 0x7B00) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;

      /* nearest rounding masks */
      unsigned short rnmask = (h.u & grs_bitmask);
      unsigned short rnmask_tie = (h.u & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = 0;	/* use a constant seed index to reuse the same set of random numbers */
	  unsigned short rand =
	    (unsigned short) _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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
      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void E5M2_DAZ_Kernel (const scalar_t * __restrict__ in,
			    scalar_t * __restrict__ out,
			    const int size,
			    const scalar_t scale,
			    int mbits, int exp_bits,
			    int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

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

    float scale_reciprocal = 1.0 / scale;
    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = in[gid] * scale;
      __half hval = __anyfloat2half_rn (inval);
      h.f = hval;
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
	  unsigned short rand = (unsigned short)
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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

      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void E4M3_Kernel (const scalar_t * __restrict__ in,
			     scalar_t * __restrict__ out,
			     const int size,
			     const scalar_t in_scale,
			     bool block_norm,
			     int mbits,
			     int exp_bits, int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

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

    extern __shared__ float sdata[];
    float scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      __float_t f;
      f.f = sdata[0];
      f.u = (f.u & 0x7F800000);
      scale = 2 * f.f;
      scale /= 8.0;
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = in[gid] * scale;

      h.f = __anyfloat2half_rn (inval);
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
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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
      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void E4M3_IEEE_Kernel (const scalar_t * __restrict__ in,
			    scalar_t * __restrict__ out,
			    const int size,
		            const scalar_t in_scale,
			    bool block_norm,
			    int mbits, int exp_bits,
			    int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

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

    extern __shared__ float sdata[];
    float scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      __float_t f;
      f.f = sdata[0];
      f.u = (f.u & 0x7F800000);
      scale = 2*f.f; 
      scale = 2 * f.f;
      scale /= 8.0;
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = in[gid] * scale;

      h.f = __anyfloat2half_rn (inval);
      short exp_h = (short) ((h.u & 0x7C00) >> 10) - 15;
      short sign_h = (h.u & 0x8000);
      short mantissa_h = (h.u & 0x03FF);

      unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      int dshift = 0;

      if (exp_h > 7) {
	/* Hardware : saturate +/-INF */
	mantissa_h = 0;
	exp_h = 16; 
	is_naninf = 1;
      } else if (exp_h < -9) {
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
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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
      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void E4M3v2_Kernel (const scalar_t *
		       __restrict__ in,
		       scalar_t * __restrict__ out,
		       const int size,
		       const scalar_t in_scale,
		       bool block_norm, int mbits,
		       int exp_bits, int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

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

    extern __shared__ float sdata[];
    float scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      __float_t f;
      f.f = sdata[0];
      f.u = (f.u & 0x7F800000);
      scale = 2 * f.f;
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = in[gid] * scale;

      h.f = __anyfloat2half_rn (inval);
      short exp_h = (short) ((h.u & 0x7C00) >> 10) - 15;
      short sign_h = (h.u & 0x8000);
      short mantissa_h = (h.u & 0x03FF);

      unsigned short can_round = ((h.u & 0x7FFF) < 0x4B80) ? 1 : 0;
      unsigned short is_normal = (((h.u & 0x7C00) <= 0x7800)
				  && ((h.u & 0x7C00) >= 0x0400)) ? 1 : 0;
      unsigned short is_denorm = ((h.u & 0x7C00) == 0x0) ? 1 : 0;
      unsigned short is_naninf = ((h.u & 0x7C00) == 0x7C00) ? 1 : 0;

      int dshift = 0;

      if (exp_h > -1) {
	/* handle saturation */
	mantissa_h = 0x0380;
	exp_h = -1;
	can_round = 0;
      }
      /* nearest rounding masks */
      unsigned short rnmask = (mantissa_h & grs_bitmask);
      unsigned short rnmask_tie = (mantissa_h & rne_tie);

      if (is_naninf == 0) {
	if (sr_mask) {
	  /* stochastic with 16 seeds */
	  int seed_index = (gid / 16);
	  unsigned short rand = (unsigned short)
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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
      mantissa_h <<= dshift;
      mantissa_h += ((exp_h + 15) << 10);
      mantissa_h |= sign_h;
      h.u = mantissa_h;
      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void E3M4_Kernel (const scalar_t * __restrict__ in,
		       scalar_t * __restrict__ out,
	               const int size,
		       const scalar_t in_scale,
		       bool block_norm,
		       int mbits,
		       int exp_bits, int rmode) {
    int non_mant_bits = exp_bits + 1;	/* exponent + sign */
    int lshift = 10 - (mbits - non_mant_bits);

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

    extern __shared__ float sdata[];
    float scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      __float_t f;

      f.f = sdata[0];
      f.u = (f.u & 0x7F800000);
      scale = 2 * f.f;
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __half_t h;
      float inval = scale * in[gid];

      h.f = __anyfloat2half_rn (inval);
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
	    _rand_xorshft128plus_with_seed (sptr[(seed_index % 16)]);
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
      __half2anyfloat (h.f * scale_reciprocal, &out[gid]);
    }
  }

  template < typename scalar_t >
    __global__ void FP4_Nearest_Kernel (const scalar_t * __restrict__ in,
		       scalar_t * __restrict__ out,
	               const int size,
		       const scalar_t in_scale,
		       bool block_norm) {
    extern __shared__ float sdata[];
    float scale = in_scale;

    if (block_norm == true) {
      absmax_block (in, sdata, size);
      /* FP4 max value is 1.0 */
      scale = 1.0/sdata[0];
    }
    float scale_reciprocal = 1.0 / scale;

    for (int gid = (blockIdx.x * blockDim.x) + threadIdx.x; gid < size;
	 gid += blockDim.x * gridDim.x) {
      __float_t f;
      float inval = scale * in[gid];

      f.f = inval;
      int exp_f = (int)((f.u & 0x7F800000) >> 23) - 127;
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
      if (exp_f < -12) f.u = 0;

      out[gid] = (f.f * scale_reciprocal);
    }
  }

}

std::vector<torch::Tensor> fpemu_cuda_forward(
    torch::Tensor input,
    std::string mode,
    const int size, 
    bool inplace, 
    float scale,
    bool block_norm,
    int block_size) { 

  float fmax = std::numeric_limits<at::Half>::max();
  if (scale > fmax) {
    fprintf(stderr,"Error: Invalid scale factor : %.2e, make sure the scale is not larger than : %.2e\n", scale, fmax);
    exit(1);
  }

  torch::Tensor output; 
  if (!inplace ) output = torch::zeros_like(input); 
 
  int sdata_size = 0;
  int threads = CUBLOCK_SIZE;
  if (block_norm == true && block_size != size) { 
	if (size%block_size) { 
		block_norm = false;
  	} else {
  		threads = block_size;
		sdata_size = threads * sizeof(float); 
	}
  }
  const dim3 blocks((size + (threads-1))/threads); 

  if (!mode.compare("E5M2_RTZ")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RTZ", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_RTZ);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());
  } else if (!mode.compare("E5M2_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RNE", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());
  } else if (!mode.compare("E5M2_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_STOCHASTIC", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());
  } else if (!mode.compare("E5M2_RNAZ")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RNAZ", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_RNAZ);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_RNTZ")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RNTZ", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_RNTZ);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_RPINF")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RPINF", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_PINF);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

   } else if (!mode.compare("E5M2_RNINF")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_RNINF", ([&] {
    	E5M2_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
		5,
		ROUND_NINF);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_DAZ_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_DAZ_RNE", ([&] {
        E5M2_DAZ_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
        	8,
		5,
		ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_DAZ_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_DAZ_STOCHASTIC", ([&] {
    	E5M2_DAZ_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
        	8,
		5,
		ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_DAZ_RNAZ")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_DAZ_RNAZ", ([&] {
    	E5M2_DAZ_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
        	8,
		5,
		ROUND_RNAZ);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E5M2_DAZ_RNTZ")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E5M2_DAZ_RNTZ", ([&] {
    	E5M2_DAZ_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
        	8,
		5,
		ROUND_RNTZ);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("FLOAT16_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "FLOAT16_RNE", ([&] {
    	FLOAT16_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		ROUND_RNE,
		0);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("FLOAT16_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "FLOAT16_STOCHASTIC", ([&] {
    	FLOAT16_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		ROUND_STOCHASTIC,
		0);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("FLOAT16_DAZ_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "FLOAT16_DAZ_RNE", ([&] {
    	FLOAT16_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		ROUND_RNE,
		1);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("BFLOAT16_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "BFLOAT16_RNE", ([&] {
    	BFLOAT16_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("BFLOAT16_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "BFLOAT16_STOCHASTIC", ([&] {
    	BFLOAT16_Kernel<scalar_t><<<blocks, threads>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E4M3_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E4M3_RNE", ([&] {
    	E4M3_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		4,
	       	ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E4M3_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E4M3_STOCHASTIC", ([&] {
    	E4M3_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		4,
	       	ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E4M3_IEEE_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E4M3_IEEE_RNE", ([&] {
    	E4M3_IEEE_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		4,
	       	ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E4M3_IEEE_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E4M3_IEEE_STOCHASTIC", ([&] {
    	E4M3_IEEE_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		4,
	       	ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E3M4_RNE")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E3M4_RNE", ([&] {
    	E3M4_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		3,
	       	ROUND_RNE);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("E3M4_STOCHASTIC")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "E3M4_STOCHASTIC", ([&] {
    	E3M4_Kernel<scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm,
        	8,
      		3,
	       	ROUND_STOCHASTIC);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());

  } else if (!mode.compare("FP4_NEAREST")) {
  	AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "FP4_NEAREST", ([&] {
    	FP4_Nearest_Kernel <scalar_t><<<blocks, threads, sdata_size>>>(
        	input.data_ptr<scalar_t>(),
  		(inplace) ? input.data_ptr<scalar_t>() : output.data_ptr<scalar_t>(),
        	size, 
		scale,
            	block_norm);
  	}));
	cudaDeviceSynchronize();
	AT_CUDA_CHECK(cudaGetLastError());
  }
  if (!inplace) {
    return {output, }; 
  } else { 
    return {input, };
  }
}
