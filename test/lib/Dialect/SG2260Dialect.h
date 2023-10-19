#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include <cstdint>

#ifndef SG2260DIALECT_H
#define SG2260DIALECT_H

#include "SG2260OpsDialect.h.inc"

namespace sg2260 {

struct DMARegDef {
  // 1024bits
  uint64_t intr_en : 1;
  uint64_t stride_enable : 1;
  uint64_t nchw_copy : 1;
  uint64_t cmd_short : 1;
  uint64_t reserved0 : 1;
  uint64_t reserved1 : 4;
  uint64_t reserved2 : 20;
  uint64_t reserved3 : 3;
  uint64_t cmd_type : 4;
  uint64_t cmd_special_function : 3;
  uint64_t fill_constant_en : 1;
  uint64_t src_data_format : 3;
  uint64_t reserved4 : 21;
  uint64_t cmd_id_dep : 24;
  uint64_t reserved5 : 8;
  uint64_t constant_value : 32;
  uint64_t src_nstride : 32;
  uint64_t src_cstride : 32;
  uint64_t src_hstride : 32;
  uint64_t src_wstride : 32;
  uint64_t dst_nstride : 32;
  uint64_t dst_cstride : 32;
  uint64_t dst_hstride : 32;
  uint64_t dst_wstride : 32;
  uint64_t src_nsize : 16;
  uint64_t src_csize : 16;
  uint64_t src_hsize : 16;
  uint64_t src_wsize : 16;
  uint64_t dst_nsize : 16;
  uint64_t dst_csize : 16;
  uint64_t dst_hsize : 16;
  uint64_t dst_wsize : 16;
  uint64_t src_start_addr_l32 : 32;
  uint64_t src_start_addr_h8 : 8;
  uint64_t reserved6 : 24;
  uint64_t dst_start_addr_l32 : 32;
  uint64_t dst_start_addr_h8 : 8;
  uint64_t reserved7 : 24;
  uint64_t reserved8 : 32;
  uint64_t reserved9 : 32;
  uint64_t localmem_mask_l32 : 32;
  uint64_t localmem_mask_h32 : 32;
  bool operator==(const DMARegDef &rhs) const { return this == &rhs; }
};

struct Matrix2RegDef {
  // 1024bits
  uint64_t cmd_short : 1;
  uint64_t op_code : 16;
  uint64_t cmd_id_dep : 24;
  uint64_t tsk_typ : 4;
  uint64_t tsk_eu_typ : 5;
  uint64_t opt_rq : 1;
  uint64_t tsk_opd_num : 2;
  uint64_t pad_mode : 2;
  uint64_t opt_res0_sign : 1;
  uint64_t rsvd0 : 3;
  uint64_t pwr_step : 4;
  uint64_t intr_en : 1;
  uint64_t opt_res_add : 1;
  uint64_t opt_relu : 1;
  uint64_t opt_left_tran : 1;
  uint64_t opt_opd4_const : 1;
  uint64_t opt_kernel_rotate : 1;
  uint64_t opt_opd0_sign : 1;
  uint64_t opt_opd1_sign : 1;
  uint64_t opt_opd2_sign : 1;
  uint64_t opt_res0_prec : 3;
  uint64_t opt_opd0_prec : 3;
  uint64_t opt_opd1_prec : 3;
  uint64_t opt_opd2_prec : 3;
  uint64_t opt_opd0_const : 1;
  uint64_t opt_opd1_const : 1;
  uint64_t opt_opd2_const : 1;
  uint64_t short_res0_str : 3;
  uint64_t short_opd0_str : 3;
  uint64_t short_opd1_str : 3;
  uint64_t short_opd2_str : 3;
  uint64_t opt_res_add_sign : 1;
  uint64_t rsvd2 : 25;
  uint64_t sym_range : 1;
  uint64_t opt_opd3_const : 1;
  uint64_t opt_opd5_const : 1;
  uint64_t opd0_x_ins0 : 4;
  uint64_t opd0_y_ins0 : 4;
  uint64_t opd1_x_ins0 : 4;
  uint64_t opd1_y_ins0 : 4;
  uint64_t opd0_up_pad : 4;
  uint64_t opd0_dn_pad : 4;
  uint64_t opd0_lf_pad : 4;
  uint64_t opd0_rt_pad : 4;
  uint64_t res_op_x_str : 4;
  uint64_t res_op_y_str : 4;
  uint64_t res0_h_shift : 4;
  uint64_t res0_w_shift : 4;
  uint64_t opd0_h_shift : 4;
  uint64_t opd0_w_shift : 4;
  uint64_t opd1_h_shift : 4;
  uint64_t opd1_w_shift : 4;
  uint64_t tsk_lane_num : 64;
  uint64_t res0_n : 16;
  uint64_t res0_c : 16;
  uint64_t res0_h : 16;
  uint64_t res0_w : 16;
  uint64_t opd0_n : 16;
  uint64_t opd0_c : 16;
  uint64_t opd0_h : 16;
  uint64_t opd0_w : 16;
  uint64_t opd1_n : 16;
  uint64_t opd1_c : 16;
  uint64_t opd1_h : 16;
  uint64_t opd1_w : 16;
  uint64_t res0_n_str : 16;
  uint64_t res0_c_str : 16;
  uint64_t opd0_n_str : 16;
  uint64_t opd0_c_str : 16;
  uint64_t opd1_n_str : 16;
  uint64_t opd1_c_str : 16;
  uint64_t opd2_n_str : 16;
  uint64_t opd2_c_str : 16;
  uint64_t res0_addr : 32;
  uint64_t opd0_addr : 32;
  uint64_t opd1_addr : 32;
  uint64_t opd2_addr : 32;
  uint64_t res0_h_str : 32;
  uint64_t res0_w_str : 32;
  uint64_t opd0_h_str : 32;
  uint64_t opd0_w_str : 32;
  uint64_t opd1_h_str : 32;
  uint64_t opd1_w_str : 32;
  uint64_t opd2_h_str : 32;
  uint64_t opd2_w_str : 32;
  uint64_t res1_addr : 32;
  uint64_t opd3_addr : 32;
  bool operator==(const Matrix2RegDef &rhs) const { return this == &rhs; }
};
} // namespace sg2260

#define GET_OP_CLASSES
#include "SG2260Ops.h.inc"

#endif
