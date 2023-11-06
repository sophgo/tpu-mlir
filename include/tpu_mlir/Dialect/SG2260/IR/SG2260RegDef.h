//===- sg2260RefDef.cpp - SG2260 register definition  ---------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

struct ShortMatrix2RegDef {
  // 384bits
  uint64_t cmd_short : 1;
  uint64_t sym_range : 1;
  uint64_t rsvd0 : 1;
  uint64_t rsvd1 : 1;
  uint64_t opt_relu : 1;
  uint64_t opt_rq : 1;
  uint64_t rsvd2 : 3;
  uint64_t rsvd3 : 8;
  uint64_t cmd_id_dep : 24;
  uint64_t tsk_typ : 4;
  uint64_t tsk_eu_typ : 5;
  uint64_t opt_res_add : 1;
  uint64_t opt_left_tran : 1;
  uint64_t opt_opd0_const : 1;
  uint64_t opt_opd2_const : 1;
  uint64_t rsvd4 : 1;
  uint64_t opt_res0_sign : 1;
  uint64_t rsvd5 : 3;
  uint64_t pwr_step : 4;
  uint64_t intr_en : 1;
  uint64_t opt_opd0_sign : 1;
  uint64_t opt_opd1_sign : 1;
  uint64_t opt_opd2_sign : 1;
  uint64_t opt_res0_prec : 3;
  uint64_t opt_opd0_prec : 3;
  uint64_t opd2_n_str : 3;
  uint64_t rsvd6 : 20;
  uint64_t rsvd7 : 16;
  uint64_t res0_c : 16;
  uint64_t res0_w : 16;
  uint64_t opd0_n : 16;
  uint64_t opd0_c : 16;
  uint64_t opd0_w : 16;
  uint64_t rsvd8 : 16;
  uint64_t opd1_w : 16;
  uint64_t res0_addr : 32;
  uint64_t opd0_addr : 32;
  uint64_t opd1_addr : 32;
  uint64_t opd2_addr : 32;
  uint64_t rsvd9 : 32;
  bool operator==(const ShortMatrix2RegDef &rhs) const { return this == &rhs; }
};
