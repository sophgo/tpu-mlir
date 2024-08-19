//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <stdint.h>

namespace tpu_mlir {

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t id;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t od;
  int64_t oh;
  int64_t ow;
  int64_t fn, fc, fh, fw;
  int64_t kd, dd, sd, ins_d;
  int64_t kh, dh, sh, ins_h;
  int64_t kw, dw, sw, ins_w;
  int64_t pdf, pdb;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t groups;
  int64_t pad_value;
  int64_t kernel_zp;
  int64_t dims; // 1d/2d/3d
  int64_t weight_is_coeff;
  bool has_bias;
  bool is_dw;
  bool do_relu;
  double relu_limit;
  int use_winograd;
  int use_3ic_optimize;
} conv_attr_t;

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t id;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t od;
  int64_t oh;
  int64_t ow;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t sd;
  int64_t sh;
  int64_t sw;
  int64_t dd;
  int64_t dh;
  int64_t dw;
  int64_t pad_d;
  int64_t pad_d_after;
  int64_t pad_h;
  int64_t pad_h_after;
  int64_t pad_w;
  int64_t pad_w_after;
  int64_t output_pad_d;
  int64_t output_pad_h;
  int64_t output_pad_w;
  int64_t pad_value;
  int64_t g;
  bool with_bias;
  bool do_relu;
  double relu_limit;
  bool is_dw;
  bool pad_insert_is_const;
} deconv_attr_t;

typedef struct {
  int64_t batch;
  int64_t M;
  int64_t K;
  int64_t N;
  int64_t batch_low;
  bool with_bias;
  bool do_relu;
  double relu_limit;
  bool left_transpose;
  bool right_transpose;
  bool output_transpose;
  bool hdim_is_batch;
  int64_t input_zp;
  int64_t right_zp;
  int64_t left_reuse;
  std::vector<int64_t> L_shape;
  std::vector<int64_t> R_shape;
  int dims_merge_2_M;
} matmul_attr_t;

typedef struct {
  int64_t n;
  int64_t c;
  int64_t id;
  int64_t ih;
  int64_t iw;
  int64_t od;
  int64_t oh;
  int64_t ow;
  int64_t kd;
  int64_t kh;
  int64_t kw;
  int64_t sd;
  int64_t sh;
  int64_t sw;
  int64_t pad_d;
  int64_t pad_d_after;
  int64_t pad_h;
  int64_t pad_h_after;
  int64_t pad_w;
  int64_t pad_w_after;
  int64_t pad_value;
  bool do_relu;
  double relu_limit;
  bool is_global;
  bool is_adaptive;
  bool count_include_pad;
  int64_t round_mode;
  int64_t src_round_mode;
} pool_attr_t;

typedef struct {
  int64_t seq_len;
  int64_t batch_size;
  int64_t input_size;
  int64_t num_direction;
  int64_t hidden_size;
  bool batch_first;
  bool have_bias;
  bool have_h0;
  bool have_c0;
  bool have_cont;
  bool output_y;
  bool output_yh;
  bool output_yc;
} lstm_attr_t;

typedef struct {
  int64_t seq_len;
  int64_t batch_size;
  int64_t input_size;
  int64_t num_direction;
  int64_t hidden_size;
  bool batch_first;
  bool have_bias;
  bool have_h0;
  bool output_y;
  bool output_yh;
} gru_attr_t;

typedef struct {
  int64_t outer_n;
  int64_t outer_c;
  int64_t axis_dims;
  int64_t inner_dims;
  bool simplified;
} reduce_attr_t;

typedef struct {
  std::vector<int64_t> in_shape_fix;
  std::vector<int64_t> out_shape_fix;
  std::vector<int64_t> order_fix;
} permute_attr_t;

typedef struct {
  std::vector<int64_t> is_4;
  std::vector<int64_t> os_4;
  std::vector<int64_t> offset_4;
  std::vector<int64_t> step_4;
  bool no_step;
  bool fusible;
} slice_attr_t;

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t id;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t od;
  int64_t oh;
  int64_t ow;
  int64_t kd, dd, sd, ins_d;
  int64_t kh, dh, sh, ins_h;
  int64_t kw, dw, sw, ins_w;
  int64_t ofc, ofd, ofh, ofw;
  int64_t mkc, mkd, mkh, mkw;
  int64_t pdf, pdb;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t groups;
  int64_t deform_groups;
  int64_t pad_value;
  int64_t kernel_zp;
  bool use_mask;
  bool has_bias;
  bool do_relu;
  bool is_dw;
  double relu_limit;
} deform_conv2d_attr_t;

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t oh;
  int64_t ow;
  int64_t kh, dh, sh;
  int64_t kw, dw, sw;
  int64_t ofc, ofh, ofw;
  int64_t mkc, mkh, mkw;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t deform_groups;
  bool use_mask;
} deform_gather_attr_t;

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t oh;
  int64_t ow;
  int64_t kh,kw;
  int64_t sh,sw;
  int64_t dh,dw;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t groups;
  bool has_bias;
} convbwd_weight_attr_t;

typedef struct {
  int64_t n;
  int64_t ic;
  int64_t ih;
  int64_t iw;
  int64_t oc;
  int64_t oh;
  int64_t ow;
  int64_t kh,kw;
  int64_t sh,sw;
  int64_t dh,dw;
  int64_t pht, phb;
  int64_t pwl, pwr;
  int64_t groups;
  int64_t insh,insw;
} convbwd_attr_t;
} // namespace tpu_mlir
