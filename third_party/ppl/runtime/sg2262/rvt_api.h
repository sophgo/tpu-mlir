#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "tpu_defs.h"

#define TCR int

#define SFU_TAYLOR_TABLE_SIZE       32
#define SFU_TAYLOR_L_TABLE_SIZE     64
#define ERF_TAYLOR_SIZE             16
#define STATIC_MEM_OFFSET           0
#define SERIAL_NUMBER_SIZE          64
#define SIN_TAYLOR_SIZE             32
#define COS_TAYLOR_SIZE             32
#define ARCSIN_TAYLOR_SIZE          64
#define TAN_TAYLOR_SIZE             32
#define POW2_LBIT_TAYLOR_SIZE       8
#define POW2_HBIT_LUT_SIZE          16
#define EXP_TAYLOR_OFFSET           (STATIC_MEM_OFFSET)
#define LOG_TAYLOR_OFFSET           (EXP_TAYLOR_OFFSET + SFU_TAYLOR_TABLE_SIZE * sizeof(float))
#define ERF_TAYLOR_OFFSET           (LOG_TAYLOR_OFFSET + SFU_TAYLOR_L_TABLE_SIZE * sizeof(float))
#define SERIAL_NUMBER_OFFSET        (ERF_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_TAYLOR_OFFSET           (SERIAL_NUMBER_OFFSET + SERIAL_NUMBER_SIZE * sizeof(float))
#define COS_TAYLOR_OFFSET           (SIN_TAYLOR_OFFSET + SIN_TAYLOR_SIZE * sizeof(float))
#define ARCSIN_TAYLOR_OFFSET        (COS_TAYLOR_OFFSET + COS_TAYLOR_SIZE * sizeof(float))
#define TAN_TAYLOR_OFFSET           (ARCSIN_TAYLOR_OFFSET + ARCSIN_TAYLOR_SIZE * sizeof(float))
#define EXP_FP16_TAYLOR_OFFSET      (TAN_TAYLOR_OFFSET + TAN_TAYLOR_SIZE * sizeof(float))
#define EXP_BF16_TAYLOR_OFFSET      (EXP_FP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(short))
#define ERF_FP16_TAYLOR_OFFSET      (EXP_BF16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(short))
#define ERF_BF16_TAYLOR_OFFSET      (ERF_FP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(short))
#define LOG_FP16_TAYLOR_OFFSET      (ERF_BF16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(short))
#define LOG_BF16_TAYLOR_OFFSET      (LOG_FP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(short))
#define SIN_FP16_TAYLOR_OFFSET      (LOG_BF16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define SIN_BFP16_TAYLOR_OFFSET     (SIN_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_FP16_TAYLOR_OFFSET      (SIN_BFP16_TAYLOR_OFFSET + ERF_TAYLOR_SIZE * sizeof(float))
#define COS_BFP16_TAYLOR_OFFSET     (COS_FP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define POW2_LBIT_TAYLOR_OFFSET     (COS_BFP16_TAYLOR_OFFSET  + ERF_TAYLOR_SIZE * sizeof(float))
#define POW2_FP16_LBIT_TAYLOR_OFFSET (POW2_LBIT_TAYLOR_OFFSET  + POW2_LBIT_TAYLOR_SIZE * sizeof(float))
#define POW2_BF16_LBIT_TAYLOR_OFFSET (POW2_FP16_LBIT_TAYLOR_OFFSET  + POW2_LBIT_TAYLOR_SIZE * sizeof(short))
#define POW2_HBIT_LUT_OFFSET        (POW2_BF16_LBIT_TAYLOR_OFFSET  + POW2_LBIT_TAYLOR_SIZE * sizeof(short))
#define POW2_FP16_HBIT_LUT_OFFSET   (POW2_HBIT_LUT_OFFSET  + POW2_HBIT_LUT_SIZE * sizeof(float))
#define POW2_BF16_HBIT_LUT_OFFSET   (POW2_FP16_HBIT_LUT_OFFSET  + POW2_HBIT_LUT_SIZE * sizeof(short))
#define SMEM_STATIC_END_OFFSET      (POW2_BF16_HBIT_LUT_OFFSET + POW2_HBIT_LUT_SIZE * sizeof(short))

typedef struct {
  union {
    int32_t dim[4];
    struct {
      uint32_t n;
      uint32_t c;
      uint32_t h;
      uint32_t w;
    };
  };
} array4_t;

typedef enum {
  TEEW_E8   = 0,
  TEEW_E16  = 1,
  TEEW_E32  = 2,
  TEEW_E4   = 5,
} TEEW_E;

typedef enum {
  TYPE_FLOAT   = 0,
  TYPE_INTEGAL  = 1,
} ITYPE_E;

typedef enum {
  HW_ALIGN_LAYOUT   = 0,
  CONTINUOUS_LAYOUT = 1,
  ROW_ALIGN_LAYOUT  = 2,
  FREE_LAYOUT       = 3
} LAYOUT_E;

typedef enum {
  INT8 = 0,
  F16 = 1, //sign=0-->fp16; sign=1-->bfp16(only for IC in atomic_gen_cmd)
  FP32 = 2,
  INT16 = 3,
  INT32 = 4,
  FP4 = 5,
  INT4 = 6,
  FP8 = 7,
  FP20 = 8,
  TF32 = 9,
  FP16 = 10,
  BFP16 = 11,
} PREC;

uint64_t rvt_sr(uint32_t rs_id, uint64_t rs);

static uint32_t __SR(uint64_t rs) {
  static uint32_t index = 0;
  uint32_t id = (index++) % 32;
  rvt_sr(id, rs);
  return id;
}

static uint16_t __prec_to_teew(int prec, bool subtype) {
  printf("----prec: %d, subtype: %d\n", prec, subtype);
  switch (prec) {
    case 0:  return (1 << 4) | (TEEW_E8 << 1) | subtype;
    case 1:  return (TEEW_E16 << 1) | subtype;
    case 2:  return (TEEW_E32 << 1) | subtype;
    case 3:  return (1 << 4) | (TEEW_E16 << 1) | subtype;
    case 4:  return (1 << 4) | (TEEW_E32 << 1) | subtype;
    case 6:  return (1 << 4) | (TEEW_E4 << 1) | subtype;
    case 7:  return (TEEW_E8 << 1) | subtype;
    case 10: return (TEEW_E16 << 1) | subtype;
    case 11: return (TEEW_E16 << 1) | 1;
    default: return 0;
  }
  return 0;
}

static uint64_t pack_scale_val(uint32_t multiplier, int shift, int offset) {
  uint64_t scale = (uint64_t)multiplier;
  scale |= (((uint64_t)shift & 0xff) << 32) | (((uint64_t)offset & 0xffff) << 40);
  return scale;
}

static inline uint64_t __pack_teew_layout(int teew, int layout) {
  return (teew << 11) | layout << 4;
}

static array4_t array4(int n, int c, int h, int w) {
  array4_t ret;
  ret.n = n;
  ret.c = c;
  ret.h = h;
  ret.w = w;
  return ret;
}

static inline uint64_t __pack_pad(int pad_mode, array4_t pads, uint32_t val) {
  uint64_t rs = val;
  rs |= ((uint64_t)pads.dim[0] & 0xf) << 32;
  rs |= ((uint64_t)pads.dim[1] & 0xf) << 36;
  rs |= ((uint64_t)pads.dim[2] & 0xf) << 40;
  rs |= ((uint64_t)pads.dim[3] & 0xf) << 44;
  rs |= ((uint64_t)pad_mode & 0xf) << 48;
  return __SR(rs);
}

static int get_sr_id() {
  return rand() % 31 + 1;
}

TCR acquire_treg(int start, int end);
void release_treg(TCR treg);

#define SIGN(dtype) ((dtype) & 0x1)
#define FP8TYPE(dtype) ((dtype) >> 5)
void rvt_csrw64(const uint32_t _idx, uint64_t rs_id);
void rvt_csrw32(const uint32_t _idx, uint64_t rs_id);
void rvt_csrw16(const uint32_t _idx, uint64_t rs_id);
void rvt_csrw8(const uint32_t _idx, uint64_t rs_id);
void rvt_sync_i(uint64_t rs_id, const uint32_t _engine);
void rvt_parallel(const uint32_t _en);
void rvt_cfgtcr(TCR trd, uint64_t rs_id, const uint32_t _update);
void rvt_cfgcr32(TCR ca, uint64_t rs_id, const uint32_t _itype, const uint32_t _teew, const uint32_t _subtype);
void rvt_cfgtcr_dim4(TCR trd, uint64_t rs_id, const uint32_t _update);
void rvt_cfgtcr_hwstride(TCR trd, uint64_t rs_id, const uint32_t _update);
void rvt_cfgtcr_ncstride(TCR trd, uint64_t rs_id, const uint32_t _update);
void rvt_cfgtcr_hwdim(TCR trd, uint64_t rs_id, const uint32_t _reset, const uint32_t _update);
void rvt_cfgtcr_ncdim(TCR trd, uint64_t rs_id, const uint32_t _reset, const uint32_t _update);
void rvt_cfgtcr_cwdim(TCR trd, uint64_t rs_id, const uint32_t _reset, const uint32_t _update);
void rvt_cfgtcr_dim1(TCR trd, uint64_t rs_id, const uint32_t _dim, const uint32_t _reset, const uint32_t _update);
void rvt_msgsend(uint64_t rs_id);
void rvt_msgwait(uint64_t rs_id);
void rvt_rand_seed(uint64_t rs_id);
void rvt_smem_bc(TCR td, uint64_t rs_id);
void rvt_smem_dist(TCR td, uint64_t rs_id);
void rvt_dma_msgsend(uint64_t rs_id);
void rvt_dma_msgwait(uint64_t rs_id);
void rvt_tgcrw32(const uint32_t _idx, uint64_t rs_id);
void rvt_csrw8_idx(const uint32_t _idx, const uint32_t _val);
void rvt_cfgtcr2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _update);
void rvt_cfgtcr_hwstride2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _update);
void rvt_cfgtcr_ncstride2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _update);
void rvt_cfgtcr_hwdim2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _reset, const uint32_t _update);
void rvt_cfgtcr_ncdim2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _reset, const uint32_t _update);
void rvt_cfgtcr_cwdim2(TCR trd, uint64_t rs1_id, uint64_t rs2_id, const uint32_t _reset, const uint32_t _update);
void rvt_end(const uint32_t _fake);
void rvt_dma_end(const uint32_t _fake);
void rvt_dma_resetid(uint64_t rs_id);
void rvt_dma_nop();

static TCR rvt_cr(int reg, int prec, bool sign, void *val) {
  TCR ca = (TCR) reg;
  uint64_t *val_ptr = (uint64_t*)val;
  uint16_t teew = __prec_to_teew(prec, sign);
  rvt_cfgtcr(ca, __SR((uint64_t)teew << 59 | ((*val_ptr) & 0xffffffffffffff)), 0);
  return ca;
}

static TCR rvt_tr(int reg, int prec, bool sign, uint32_t offset,
        LAYOUT_E layout, array4_t shape, int *strides) {
  TCR ta = (TCR) reg;
  uint16_t teew = __prec_to_teew(prec, sign);
  rvt_cfgtcr(ta, __SR((__pack_teew_layout(teew, layout) << 48) | offset), 0);
  uint64_t dim = (((uint64_t)shape.n & 0xffff) << 48) |
                  (((uint64_t)shape.c & 0xffff) << 32) |
                  (((uint64_t)shape.h & 0xffff) << 16) |
                   ((uint64_t)shape.w & 0xffff);
  rvt_cfgtcr_dim4(ta, __SR(dim), 1);
  if (strides && layout == FREE_LAYOUT) {
    rvt_cfgtcr_hwstride(ta, __SR((uint64_t)strides[2] << 32 | strides[3]), 0);
    rvt_cfgtcr_ncstride(ta, __SR((uint64_t)strides[0] << 32 | strides[1]), 1);
  }
  return ta;
}

static inline void rvt_cfg_pad(int pad_mode, array4_t pads, void *val) {
  uint64_t rs = __pack_pad(pad_mode, pads, *((uint32_t*)val));
  rvt_csrw64(0x4, rs);
}

static inline uint64_t __pack_insrt(int ins_i_h, int ins_i_w, int ins_k_h, int ins_k_w, uint32_t val) {
  uint64_t rs = val;
  rs |= ((uint64_t)ins_i_w & 0xf) << 32;
  rs |= ((uint64_t)ins_i_h & 0xf) << 36;
  rs |= ((uint64_t)ins_k_w & 0xf) << 40;
  rs |= ((uint64_t)ins_k_h & 0xf) << 44;
  return __SR(rs);
}

static inline void rvt_cfg_insrt(int ins_i_h, int ins_i_w, int ins_k_h, int ins_k_w, void *val) {
  uint64_t rs = __pack_insrt(ins_i_h, ins_i_w, ins_k_h, ins_k_w, *((uint32_t*)val));
  rvt_csrw64(0x5, rs);
}

static inline uint64_t __pack_stenceil(int kh, int kw, int sh, int sw, bool krotate, bool do_relu) {
  uint64_t rs = (uint64_t)kw & 0xffff;
  rs |= ((uint64_t)kh & 0xffff) << 16;
  rs |= ((uint64_t)sw & 0xff) << 32;
  rs |= ((uint64_t)sh & 0xff) << 40;
  rs |= ((uint64_t)krotate  & 0x1) << 62;
  rs |= ((uint64_t)do_relu  & 0x1) << 63;
  return __SR(rs);
}

static inline void rvt_cfg_stencil(int kh, int kw, int sh, int sw, bool krotate, bool do_relu) {
  uint64_t rs = __pack_stenceil(kh, kw, sh, sw, krotate, do_relu);
  rvt_csrw64(0x6, rs);
}

static inline void rvt_cfg_lanemask(uint64_t lane_mask) {
  (void)lane_mask;
}

static void rvt_cfg_round_mode(int round_mode) {
  rvt_csrw8(0, __SR(round_mode));
}

static void rvt_cfg_satu(uint8_t sym_satu, uint8_t f_satu) {
  (void)sym_satu;
  rvt_csrw8(1, __SR(f_satu));
}

static void rvt_cfg_tf32(uint8_t is_tf32) {
  rvt_csrw8(2, __SR(is_tf32));
}

static void rvt_cfg_rsqrt_iter(uint8_t rsqrt_iter) {
  rvt_csrw8(3, __SR(rsqrt_iter));
}

static void rvt_cfg_quant(TCR ts) {
  rvt_csrw8(7, ts);
}

static void rvt_cfg_kzp(TCR ts) {
  // not support in 2262
}

static void rvt_cfg_pwrstep(uint8_t pwr_step) {
  rvt_csrw8(12, __SR(pwr_step));
}

static void rvt_cfg_jump_cnt(uint16_t jump_cnt) {
  rvt_csrw16(2, __SR(jump_cnt));
}

static inline TCR ca0() {
  return 0;
}

static TCR rvt_gr(int reg, int prec, bool sign, uint64_t addr,
        LAYOUT_E layout, array4_t shape, int *strides) {
  TCR ga = (TCR) reg;
  uint16_t teew = __prec_to_teew(prec, sign);
  rvt_cfgtcr(ga, __SR((__pack_teew_layout(teew, layout) << 48) | addr), 0);
  if (shape.h < (1 << 16) && shape.w < (1 << 16)) {
    uint64_t size = (((uint64_t)shape.n & 0xffff) << 48) |
                    (((uint64_t)shape.c & 0xffff) << 32) |
                    (((uint64_t)shape.h & 0xffff) << 16) |
                    ((uint64_t)shape.w & 0xffff);
    rvt_cfgtcr_dim4(ga, __SR(size), 1);
  } else {
    uint64_t hw_size = (((uint64_t)shape.h & 0xffffffff) << 32) |
                       (((uint64_t)shape.w & 0xffffffff));
    uint64_t nc_size = (((uint64_t)shape.n & 0xffff) << 16) |
                       (((uint64_t)shape.c & 0xffff));
    rvt_cfgtcr_hwdim(ga, __SR(hw_size), 0, 0);
    rvt_cfgtcr_ncdim(ga, __SR(nc_size), 0, 1);
  }

  if (strides) {
    assert(layout == FREE_LAYOUT);
    rvt_cfgtcr_hwstride(ga, __SR((uint64_t)strides[2] << 32 | (uint64_t)strides[3]), 0);
    rvt_cfgtcr_ncstride(ga, __SR((uint64_t)strides[0] << 32 | (uint64_t)strides[1]), 1);
  }
  return ga;
}
static PREC PRECISION(data_type_t dtype) {
    switch (dtype)
    {
    case DT_FP32:
        return FP32;
    case DT_FP16:
        return FP16;
    case DT_INT8:
    case DT_UINT8:
        return INT8;
    case DT_INT16:
    case DT_UINT16:
        return INT16;
    case DT_INT32:
    case DT_UINT32:
        return INT32;
    case DT_INT4:
    case DT_UINT4:
        return INT4;
    case DT_BFP16:
        return BFP16;
    case DT_FP8E5M2:
    case DT_FP8E4M3:
        return FP8;
    case DT_TF32:
        return TF32;
    default:
        assert(0 && "Invalid data type");
        break;
    }
    return FP32;
}

void rvt_cmp_max_gt(TCR out_gt, TCR out_max, TCR a, TCR b, TCR c, TCR d);
void rvt_cmp_min_lt(TCR out_lt, TCR out_min, TCR a, TCR b, TCR c, TCR d);
void rvt_cmplt(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_fcmplt(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_cmpeq(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_fcmpeq(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_cmpgt(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_fcmpgt(TCR out, TCR a, TCR b, TCR c, TCR d);
void rvt_fconv(TCR out, TCR x, TCR w, TCR bias, TCR rq);
void rvt_fconva(TCR out, TCR x, TCR w, TCR bias, TCR rq);
void rvt_dwconv(TCR out, TCR x, TCR w, TCR bias, TCR rq);
void rvt_fdwconv(TCR out, TCR x, TCR w, TCR bias, TCR rq);
void rvt_pool_avg(TCR out, TCR x, TCR w, TCR rq);
void rvt_pool_favg(TCR out, TCR x, TCR w, TCR rq);
void rvt_pool_max(TCR out, TCR x);
void rvt_pool_fmax(TCR out, TCR x);
void rvt_pool_min(TCR out, TCR x);
void rvt_pool_fmin(TCR out, TCR x);
void rvt_fconvdw(TCR out, TCR x, TCR dy);
void rvt_fmm2_nn(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2_nt(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2_tt(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2a_nn(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2a_nt(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2a_tt(TCR out, TCR x, TCR w, TCR bias, TCR rq, const uint32_t _relu);
void rvt_fmm2_dq2_nn(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2_dq2_nt(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2_dq2_tt(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2a_dq2_nn(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2a_dq2_nt(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2a_dq2_tt(TCR out, TCR x, TCR w, TCR ws, TCR os, const uint32_t _gsize);
void rvt_fmm2_bq_nt(TCR out, TCR x, TCR w, TCR xs, TCR ws, const uint32_t _gsize);
void rvt_fmm2a_bq_nt(TCR out, TCR x, TCR w, TCR xs, TCR ws, const uint32_t _gsize);
void rvt_roipool_avg(TCR out, TCR x, TCR w, TCR roi);
void rvt_roipool_favg(TCR out, TCR x, TCR w, TCR roi);
void rvt_roipool_max(TCR out, TCR x, TCR roi);
void rvt_roipool_fmax(TCR out, TCR x, TCR roi);
void rvt_roipool_min(TCR out, TCR x, TCR roi);
void rvt_roipool_fmin(TCR out, TCR x, TCR roi);
void rvt_add(TCR out, TCR a, TCR b, TCR shift, const uint32_t _satu);
void rvt_sub(TCR out, TCR a, TCR b, TCR shift, const uint32_t _satu);
void rvt_mul(TCR out, TCR a, TCR b, TCR shift, const uint32_t _satu);
void rvt_mac(TCR out, TCR a, TCR b, TCR shift);
void rvt_abs(TCR out, TCR a);
void rvt_max(TCR out, TCR a, TCR b);
void rvt_min(TCR out, TCR a, TCR b);
void rvt_sel_gt(TCR out, TCR a, TCR b, TCR c);
void rvt_sel_eq(TCR out, TCR a, TCR b, TCR c);
void rvt_sel_lt(TCR out, TCR a, TCR b, TCR c);
void rvt_lshr(TCR out, TCR a, TCR shift);
void rvt_ashr(TCR out, TCR a, TCR shift);
void rvt_rshr(TCR out, TCR a, TCR shift);
void rvt_fadd(TCR out, TCR a, TCR b);
void rvt_fsub(TCR out, TCR a, TCR b);
void rvt_fmul(TCR out, TCR a, TCR b);
void rvt_fmac(TCR out, TCR a, TCR b);
void rvt_fdiv(TCR out, TCR a, TCR b);
void rvt_fabs(TCR out, TCR a);
void rvt_fsubabs(TCR out, TCR a, TCR b);
void rvt_fmax(TCR out, TCR a, TCR b);
void rvt_fmin(TCR out, TCR a, TCR b);
void rvt_fsel_gt(TCR out, TCR a, TCR b, TCR c);
void rvt_fsel_eq(TCR out, TCR a, TCR b, TCR c);
void rvt_fsel_lt(TCR out, TCR a, TCR b, TCR c);
void rvt_fdiv2(TCR out, TCR a, TCR b);
void rvt_cvt_i2i(TCR out, TCR a);
void rvt_cvt_i2f(TCR out, TCR a);
void rvt_cvt_f2i(TCR out, TCR a);
void rvt_cvt_f2f(TCR out, TCR a);
void rvt_clz(TCR out, TCR a);
void rvt_clo(TCR out, TCR a);
void rvt_cp(TCR out, TCR a);
void rvt_not(TCR out, TCR a);
void rvt_and(TCR out, TCR a, TCR b);
void rvt_xor(TCR out, TCR a, TCR b);
void rvt_or(TCR out, TCR a, TCR b);
void rvt_clamp(TCR out, TCR a, TCR min_v, TCR max_v);
void rvt_fmulcast(TCR out, TCR a, TCR b);
void rvt_fvcmax(TCR out, TCR a, TCR b);
void rvt_fvcmin(TCR out, TCR a, TCR b);
void rvt_vcmax(TCR out, TCR a, TCR b);
void rvt_vcmin(TCR out, TCR a, TCR b);
void rvt_fmuladd(TCR out, TCR a, TCR b, TCR c);
void rvt_fsqradd(TCR out, TCR a, TCR b);
void rvt_fsqrsub(TCR out, TCR a, TCR b);
void rvt_faxpy(TCR out, TCR a, TCR b, TCR c);
void rvt_dq0(TCR out, TCR a, TCR scale);
void rvt_rq1(TCR out, TCR a, TCR scale);
void rvt_dq2(TCR out, TCR a, TCR scale, const uint32_t _gsize);
void rvt_qt0(TCR out_scale, TCR out, TCR a, TCR range_val, TCR r_range_val);
void rvt_sfu_norm(TCR out, TCR a);
void rvt_sfu_taylor(TCR out, TCR a, TCR coeff);
void rvt_sfu_rsqrt(TCR out, TCR a);
void rvt_sfu_frexp(TCR out_frac, TCR out_exp, TCR a);
void rvt_sfu_rcpr(TCR out, TCR a);
void rvt_sfu_exp(TCR out_pld, TCR out, TCR a, TCR bias, TCR coeff);
void rvt_gather_pc(TCR out, TCR a, TCR idx, TCR cs);
void rvt_scatter_pc(TCR out, TCR a, TCR idx);
void rvt_gather2d_pc(TCR out, TCR a, TCR idx, TCR cs);
void rvt_scatter2d_pc(TCR out, TCR a, TCR idx);
void rvt_gather(TCR out, TCR a, TCR idx, TCR cs, const uint32_t _hzd);
void rvt_scatter(TCR out, TCR a, TCR idx, const uint32_t _hzd);
void rvt_hgather(TCR out, TCR a, TCR idx, TCR cs);
void rvt_hscatter(TCR out, TCR a, TCR idx);
void rvt_masksel(TCR out_cnt, TCR out, TCR a, TCR mask, const uint32_t _hzd);
void rvt_nzidx(TCR out_cnt, TCR out, TCR a, const uint32_t _hzd);
void rvt_trans_cw(TCR out, TCR a);
void rvt_trans_wc(TCR out, TCR a);
void rvt_lane_cp(TCR out, TCR a);
void rvt_lane_bc(TCR out, TCR a);
void rvt_srchbin(TCR out, TCR a, TCR b, const uint32_t _side);
void rvt_fsrchbin(TCR out, TCR a, TCR b, const uint32_t _side);
void rvt_rand0(TCR out, TCR out_state, TCR coffset);
void rvt_rand1(TCR out, TCR out_state, TCR in_state);
void rvt_reduce_max(TCR out, TCR a);
void rvt_reduce_min(TCR out, TCR a);
void rvt_reduce_amax(TCR out, TCR a);
void rvt_reduce_sum(TCR out, TCR a);
void rvt_reduce_powersum(TCR out, TCR a);
void rvt_dma_ld(TCR td, TCR src);
void rvt_dma_ldt(TCR td, TCR gs);
void rvt_dma_st(TCR gd, TCR ts);
void rvt_dma_stt(TCR gd, TCR ts);
void rvt_dma_cp(TCR dst, TCR src);
void rvt_dma_nctrans(TCR dst, TCR src);
void rvt_dma_mld(TCR td, TCR gs);
void rvt_dma_mst(TCR gd, TCR ts);
void rvt_dma_ldbc(TCR td, TCR src);
void rvt_dma_scp(TCR gd, TCR gs);
void rvt_dma_sbc(TCR td, TCR src);
void rvt_dma_red(TCR gd, TCR src, const uint32_t _rw, const uint32_t _op);
void rvt_dma_fred(TCR gd, TCR src, const uint32_t _rw, const uint32_t _op);
void rvt_dma_reverse(TCR dst, TCR src, const uint32_t _dim);
void rvt_dma_masksel(TCR dst, TCR src, TCR mask);
void rvt_dma_fmasksel(TCR dst, TCR src, TCR mask);
void rvt_dma_nzidx(TCR dst, TCR src, TCR start_pos);
void rvt_dma_fnzidx(TCR dst, TCR src, TCR start_pos);
void rvt_dma_hgather(TCR dst, TCR src, TCR idx, TCR start_pos, TCR const_val);
void rvt_dma_cgather(TCR dst, TCR src, TCR idx, TCR start_pos, TCR const_val);
void rvt_dma_hscatter(TCR dst, TCR src, TCR idx, TCR start_pos, const uint32_t _inplace_add);
void rvt_dma_cscatter(TCR dst, TCR src, TCR idx, TCR start_pos, const uint32_t _inplace_add);

static void rvt_cfgcr(TCR ca, uint64_t rs) {
  rvt_cfgtcr(ca, __SR(rs), 0);
}
static void rvt_cfgtr(TCR ta, uint64_t rs) {
  rvt_cfgtcr(ta, __SR(rs), 1);
}
static void rvt_cfgtr_shape(uint64_t ta, uint64_t rs) {
  rvt_cfgtcr_dim4(ta, __SR(rs), 0x1);
}
static void rvt_cfgtr_hwstride(uint64_t ta, uint64_t rs) {
  rvt_cfgtcr_hwstride(ta, __SR(rs), 0);
}
static void rvt_cfgtr_ncstride(uint64_t ta, uint64_t rs) {
  rvt_cfgtcr_ncstride(ta, __SR(rs), 0);
}
static void rvt_cfggr(uint64_t ga, uint64_t rs) {
  rvt_cfgtcr(ga, __SR(rs), 1);
}
static void rvt_cfggr_shape(uint64_t ga, uint64_t rs){
  rvt_cfgtcr_dim4(ga, __SR(rs), 0x1);
}
static void rvt_cfggr_hwstride(uint64_t ga, uint64_t rs) {
  rvt_cfgtcr_hwstride(ga, __SR(rs), 0);
}
static void rvt_cfggr_ncstride(uint64_t ga, uint64_t rs) {
  rvt_cfgtcr_ncstride(ga, __SR(rs), 0);
}
static void rvt_cfggr_hwdim(uint64_t ga, uint64_t rs) {
  rvt_cfgtcr_hwdim(ga, __SR(rs), 0, 0x1);
}
static void rvt_cfggr_ncdim(uint64_t ga, uint64_t rs) {
  rvt_cfgtcr_ncdim(ga, __SR(rs), 0, 0x1);
}
