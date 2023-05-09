//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "stdint.h"
#include <iostream>
#include <vector>
#include <math.h>

namespace tpu_mlir {
namespace backend {
#define DATA_MAX_DISTANCE 16
#define MAX_BURST_LENGTH 16
#define SPECIAL_FUNCTION_BURST_LENGTH 8
#define AXI_BUS_WIDTH 16   // 16Byte
#define LOCAL_MEM_WIDTH 16 // 16Byte
#define FOUR_KB 0x1000
#define FOUR_KB_MASK 0xfffff000
#define MAX_PACKET_CAPACITY MAX_BURST_LENGTH *AXI_BUS_WIDTH
#define SRC_BASE_ADDR_HIGH_SHIFT 32
#define BYTE64 0x40
#define BYTE64_MASK 0xffffffc0
#define STORE_DATA_MAX_DISTANCE 0
#define LOCAL_MEM_START_ADDR 0x00000000
#define TPU_LMEM_ADDR_WIDTH 15

#define MAGIC_CV183X 0xA5
#define MAGIC_CV182X 0xA6
#define MAGIC_CV181X 0xA7
#define MAGIC_CV180X 0xA8

typedef struct __cmd_hdr_s {
  unsigned char magic;         // 0xA5
  unsigned char len;           // lens in bytes
  unsigned char engine_id : 4; // TPU, GDMA, CDMA
  unsigned char __deprecated : 4;
  unsigned char flags; // CMD_ID, sync flags, etc. TBD
  unsigned int mask;   // bit mask for which register need to write
  unsigned char cmd[0];
} cmd_hdr_t;

enum Des_tsk_typ { Conv2D, Pooling, MatrixMul, TensorArithmetic, MatrixMul2 };

typedef struct {
  uint32_t cmd_en;
  uint32_t cmd_end;
  uint32_t cmd_id_en;
  uint32_t cmd_id_tpu;
  uint32_t cmd_id_gdma;
  uint32_t cmd_keep;
  uint32_t cmd_intr_en;
  uint32_t tsk_typ;
  uint32_t tsk_eu_typ;
  uint32_t tsk_opd_num;
  uint32_t opt_right_shift;
  uint32_t opt_left_shift;
  uint32_t opt_shift_typ;
  uint32_t opt_rshift_typ;
  uint32_t opt_res_add;
  uint32_t opt_relu;
  uint32_t opt_left_tran;
  uint32_t opt_chl_quan;
  uint32_t tens_mdsum;
  uint32_t tens_lookup;
  uint32_t opt_res0_sign;
  uint32_t opt_opd0_sign;
  uint32_t opt_opd1_sign;
  uint32_t opt_opd2_sign;
  uint32_t opt_res0_int8;
  uint32_t opt_opd0_int8;
  uint32_t opt_opd1_int8;
  uint32_t opt_opd2_int8;
  uint32_t opt_opd0_const;
  uint32_t opt_opd1_const;
  uint32_t opt_opd2_const;
  uint32_t short_nchwstr_same;
  uint32_t short_res0_str;
  uint32_t short_opd0_str;
  uint32_t short_opd1_str;
  uint32_t short_opd2_str;
  uint32_t conv_opd0_x_ins0;
  uint32_t conv_opd0_y_ins0;
  uint32_t conv_opd0_x_ins0_last;
  uint32_t conv_opd0_y_ins0_last;
  uint32_t conv_opd1_x_ins0;
  uint32_t conv_opd1_y_ins0;
  uint32_t opd0_ins_val;
  uint32_t ps32_md;
  uint32_t double_conv;
  uint32_t rsvd0;
  uint32_t res0_n;
  uint32_t res0_c;
  uint32_t res0_h;
  uint32_t res0_w;
  uint32_t res0_addr;
  uint32_t opd0_addr;
  uint32_t opd1_addr;
  uint32_t rsvd1;
  uint32_t opd2_addr;
  uint32_t opd0_c;
  uint32_t opd0_h;
  uint32_t opd0_w;
  uint32_t opd1_h;
  uint32_t opd1_w;
  uint32_t conv_opd0_up_pad;
  uint32_t conv_opd0_dn_pad;
  uint32_t conv_opd0_lf_pad;
  uint32_t conv_opd0_rt_pad;
  uint32_t conv_op_x_str;
  uint32_t conv_op_y_str;
  uint32_t opd0_ins_fp;
  uint32_t rsvd2;
  uint32_t opd0_n;
  uint32_t opd1_n;
  uint32_t opd1_c;
  uint32_t opd2_n;
  uint32_t opd2_c;
  uint32_t opd2_h;
  uint32_t opd2_w;
  uint32_t quan_m;
  uint32_t opd_typ;
  uint32_t fp_round_typ;
  uint32_t rsvd7;
  uint32_t rsvd3;
  uint32_t res0_n_str;
  uint32_t res0_c_str;
  uint32_t res0_h_str;
  uint32_t res0_w_str;
  uint32_t res0_b_str;
  uint32_t opd0_n_str;
  uint32_t opd0_c_str;
  uint32_t rsvd4;
  uint32_t opd0_h_str;
  uint32_t opd0_w_str;
  uint32_t opd0_b_str;
  uint32_t opd1_n_str;
  uint32_t opd1_c_str;
  uint32_t opd1_h_str;
  uint32_t opd1_w_str;
  uint32_t rsvd5;
  uint32_t opd1_b_str;
  uint32_t opd2_n_str;
  uint32_t opd2_c_str;
  uint32_t opd2_h_str;
  uint32_t opd2_w_str;
  uint32_t opd2_b_str;
  uint32_t layer_info;
  uint32_t rsvd6;
  ///////////////////Added in bm1822
  uint32_t cmd_pre_exe;
  uint32_t opt_res0_seg;
  uint32_t opt_opd0_seg;
  uint32_t opt_opd1_seg;
  uint32_t opt_opd2_seg;
  uint32_t opt_relu_typ;
  uint32_t opt_relu_value;
  uint32_t opt_res_shift;
  uint32_t cmd_pre_exe_typ;
} tiu_reg_t;

enum TdmaDesTskType {
  CommonDMA,
  NHWC3,
  NHWC4,
  MatrixTranspose,
  MatrixLoad,
  MatrixStore,
  CWTranspose,
  NCTranspose,
  LMEMShift,
  ConstantFilling,
  DeCompression,
  Compression,
  TensorLoad,
  TensorStore,
  TensorS2S,
  TensorL2L,
  TensorReshape
};

typedef struct {
  uint32_t vld;
  uint32_t compress_en;
  uint32_t eod;
  uint32_t intp_en;
  uint32_t bar_en;
  uint32_t check_bf16_value;
  uint32_t trans_dir;
  uint32_t rsv00;
  uint32_t trans_fmt;
  uint32_t transpose_md;
  uint32_t rsv01;
  uint32_t outstanding_en;
  uint32_t cmd_id;
  uint32_t spec_func;
  uint32_t dst_fmt;
  uint32_t src_fmt;
  uint32_t cmprs_fmt;
  uint32_t sys_dtype;
  uint32_t rsv2_1;
  uint32_t int8_sign;
  uint32_t compress_zero_guard;
  uint32_t int8_rnd_mode;
  uint32_t wait_id_tpu;
  uint32_t wait_id_other_tdma;
  uint32_t wait_id_sdma;
  uint32_t const_val;
  uint32_t src_base_reg_sel;
  uint32_t mv_lut_idx;
  uint32_t dst_base_reg_sel;
  uint32_t mv_lut_base;
  uint32_t rsv4_5;
  uint32_t dst_h_stride;
  uint32_t dst_c_stride_low;
  uint32_t dst_n_stride;
  uint32_t src_h_stride;
  uint32_t src_c_stride_low;
  uint32_t src_n_stride;
  uint32_t dst_c;
  uint32_t src_c;
  uint32_t dst_w;
  uint32_t dst_h;
  uint32_t src_w;
  uint32_t src_h;
  uint32_t dst_base_addr_low;
  uint32_t src_base_addr_low;
  uint32_t src_n;
  uint32_t dst_base_addr_high;
  uint32_t src_base_addr_high;
  uint32_t src_c_stride_high;
  uint32_t dst_c_stride_high;
  uint32_t compress_bias0;
  uint32_t compress_bias1;
  uint32_t layer_ID;

  // Add in bm1822
  uint32_t intra_cmd_paral;

} tdma_reg_t;

struct Tuple4D {
  union {
    uint64_t batch;
    uint64_t n;
    uint64_t N;
  };

  union {
    uint64_t h;
    uint64_t H;
    uint64_t height;
    uint64_t y;
    uint64_t Y;
  };

  union {
    uint64_t w;
    uint64_t W;
    uint64_t width;
    uint64_t x;
    uint64_t X;
  };

  union {
    uint64_t c;
    uint64_t C;
    uint64_t channel;
    uint64_t z;
    uint64_t Z;
  };

  Tuple4D(uint64_t _n, uint64_t _h, uint64_t _w, uint64_t _c)
      : N(_n), H(_h), W(_w), C(_c) {}

  Tuple4D() : N(1), H(0), W(0), C(0) {}

  inline void reset() {
    this->n = 0;
    this->h = 0;
    this->w = 0;
    this->c = 0;
  }

  inline uint64_t size() { return this->n * this->h * this->w * this->c; }

  inline Tuple4D &operator=(const Tuple4D &rOperand) {
    this->n = rOperand.n;
    this->h = rOperand.h;
    this->w = rOperand.w;
    this->c = rOperand.c;
    return *this;
  }

  inline Tuple4D operator+(const Tuple4D &rOperand) {
    Tuple4D ret;
    ret.n = this->n + rOperand.n;
    ret.h = this->h + rOperand.h;
    ret.w = this->w + rOperand.w;
    ret.c = this->c + rOperand.c;
    return ret;
  }

  inline Tuple4D &operator+=(const Tuple4D &rOperand) {
    this->n = this->n + rOperand.n;
    this->h = this->h + rOperand.h;
    this->w = this->w + rOperand.w;
    this->c = this->c + rOperand.c;
    return *this;
  }

  inline Tuple4D operator-(const Tuple4D &rOperand) {
    Tuple4D ret;
    ret.n = this->n - rOperand.n;
    ret.h = this->h - rOperand.h;
    ret.w = this->w - rOperand.w;
    ret.c = this->c - rOperand.c;
    return ret;
  }

  inline Tuple4D &operator-=(const Tuple4D &rOperand) {
    this->n = this->n - rOperand.n;
    this->h = this->h - rOperand.h;
    this->w = this->w - rOperand.w;
    this->c = this->c - rOperand.c;
    return *this;
  }

  inline Tuple4D operator*(const Tuple4D &rOperand) {
    Tuple4D ret;
    ret.n = this->n * rOperand.n;
    ret.h = this->h * rOperand.h;
    ret.w = this->w * rOperand.w;
    ret.c = this->c * rOperand.c;
    return ret;
  }

  inline Tuple4D &operator*=(const Tuple4D &rOperand) {
    this->n = this->n * rOperand.n;
    this->h = this->h * rOperand.h;
    this->w = this->w * rOperand.w;
    this->c = this->c * rOperand.c;
    return *this;
  }

  inline Tuple4D operator/(const Tuple4D &rOperand) {
    Tuple4D ret;
    ret.n = ceil((float)this->n / (float)rOperand.n);
    ret.h = ceil((float)this->h / (float)rOperand.h);
    ret.w = ceil((float)this->w / (float)rOperand.w);
    ret.c = ceil((float)this->c / (float)rOperand.c);
    return ret;
  }

  inline Tuple4D &operator/=(const Tuple4D &rOperand) {
    this->n = ceil((float)this->n / (float)rOperand.n);
    this->h = ceil((float)this->h / (float)rOperand.h);
    this->w = ceil((float)this->w / (float)rOperand.w);
    this->c = ceil((float)this->c / (float)rOperand.c);
    return *this;
  }

  inline bool operator==(const Tuple4D &rOperand) {
    return (this->n == rOperand.n) && (this->h == rOperand.h) &&
           (this->w == rOperand.w) && (this->c == rOperand.c);
  }

  inline bool operator!=(const Tuple4D &rOperand) {
    return !((this->n == rOperand.n) && (this->h == rOperand.h) &&
             (this->w == rOperand.w) && (this->c == rOperand.c));
  }

  inline bool operator<(const Tuple4D &rOperand) {
    return ((this->n < rOperand.n) && (this->h < rOperand.h) &&
            (this->w < rOperand.w) && (this->c < rOperand.c));
  }

  inline bool operator<=(const Tuple4D &rOperand) {
    return ((this->n <= rOperand.n) && (this->h <= rOperand.h) &&
            (this->w <= rOperand.w) && (this->c <= rOperand.c));
  }

  friend std::ostream &operator<<(std::ostream &out, const Tuple4D &operand) {
    out << "(N = " << operand.n << ", H = " << operand.h
        << ", W = " << operand.w << ", C = " << operand.c << ")";
    return out;
  }
};

class TiuReg final {
public:
  static uint64_t calCycle(tiu_reg_t &task, uint64_t tpu_frequency);
  static void parse_tiu_reg(tiu_reg_t *r, const uint32_t *p,
                            unsigned char magicNum);

private:
  static uint64_t calTiuCycle(tiu_reg_t &task);

  static float getEltwiseLatency(int taskType, bool is8BitMode,
                                 bool isOpd1Const, int mode);
  static void cv182xMapToCv183x(tiu_reg_t *r);

  static void parse_cv182x_tiu_reg(tiu_reg_t *r, const uint32_t *p);

  static void parse_cv183x_tiu_reg(tiu_reg_t *r, const uint32_t *p);

  static int getTensorArithmeticMode(int taskType, bool is8BitMode);
};

class TdmaReg final {
public:
  static uint64_t calCycle(tdma_reg_t task, uint64_t dram_frequency,
                           uint64_t sram_frequency);

  static void parse_tdma_reg(tdma_reg_t *r, const uint32_t *p,
                             unsigned char magicNum);

private:
  static void calLoad(tdma_reg_t task, uint64_t &dram_count,
                      uint64_t &sram_count);

  // copy from TdmaStorer.cc
  static void calStore(tdma_reg_t &task, uint64_t &dram_count,
                       uint64_t &sram_count);
  static uint64_t calMove(tdma_reg_t &task);

  static uint64_t inline get_src_address(tdma_reg_t &r);

  static uint64_t inline get_dst_address(tdma_reg_t &r);

  static uint64_t get_tdma_cycle(tdma_reg_t &task, uint64_t baseAddr,
                                 uint64_t data_size, bool isStore);
  static uint64_t calByteCnt(uint64_t baseAddr, uint64_t size);

  static uint64_t calSramCycle(tdma_reg_t &task);

  static void parse_cv182x_tdma_reg(tdma_reg_t *r, const uint32_t *p);

  static void parse_cv183x_tdma_reg(tdma_reg_t *r, const uint32_t *p);
};

class CV18xxProfiling final {
public:
  static uint64_t get_cycle(std::vector<uint8_t> &cmdbuf);
};

} // namespace backend
} // namespace tpu_mlir
