//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

#define DYN_BMM_CONDITION(param, spec)                                         \
  (((param.hdim_is_batch)) || ((param.batch != 1)) ||                          \
   (param.batch == 1 && spec->at(0).shape[0] == 1 &&                           \
    spec->at(1).shape[0] == 1))

void tpu::MatMulOp::codegen_global_bm1684x() {
  auto p = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto odtype = module::getStorageType(getOutput());

  if (p.hdim_is_batch || p.batch != 1 || getFuseRq() || module::isF8Modes() ||
      module::isMARS3() || module::isSGTPUV8()) {
    if (!p.hdim_is_batch) {
      BM168x::fix_shape(input_spec->at(0), {p.batch, p.M, p.K});
      if (p.right_transpose == false) {
        BM168x::fix_shape(input_spec->at(1), {p.batch, p.K, p.N});
      } else {
        BM168x::fix_shape(input_spec->at(1), {p.batch, p.N, p.K});
      }
      BM168x::fix_shape(output_spec->at(0), {p.batch, p.M, p.N});
    }
    batch_matmul_common_spec_t spec{0};
    spec.Y_dtype = output_spec->at(0).dtype;
    spec.L_trans = p.left_transpose;
    spec.R_trans = p.right_transpose;
    spec.has_bias = p.with_bias;
    spec.hdim_is_batch = p.hdim_is_batch;
    spec.requant_mode = -1;
    spec.do_relu = p.do_relu;
    spec.upper_limit = p.relu_limit;
    if (module::isUniformQuantized(getInput())) {
      spec.R_zp_is_const = true;
      spec.R_zp_const_val = p.right_zp;
      spec.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(getOutput())) {
        spec.requant_mode = static_cast<int>(getQuantMode());
        auto rshift_v = module::getI64Array(getRshifts());
        auto multiplier_v = module::getI64Array(getMultipliers());
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(getOutput());
        spec.offset_val = output_type.getZeroPoint();
        spec.fuse_rq = getFuseRq();
        if (spec.fuse_rq)
          spec.round_mode = (RoundingMode)getRoundMode();
      }
    } else if (odtype.isFloat8E4M3FN()) {
      spec.requant_mode = 0;
      f64_array_t scales = module::getF64Array(getOutF8Scales().value());
      float scale = scales->at(0);
      spec.mul_val = *(int *)&scale;
    }

    BM168x::call_global_func("backend_api_batch_matmul_global", &spec,
                             sizeof(spec), input_spec->data(),
                             output_spec->data());
    return;
  }
  BM168x::fix_shape(input_spec->at(0), {p.M, p.K});
  BM168x::fix_shape(input_spec->at(1), {p.K, p.N});
  BM168x::fix_shape(output_spec->at(0), {p.M, p.N});
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.if_relu = p.do_relu;
  spec.if_getting_buffer_size = false;
  spec.buffer_size_ptr = NULL;
  spec.relu_limit = p.relu_limit;
  spec.have_bias = p.with_bias;
  spec.requant_mode = -1;
  spec.R_transpose = p.N == 1 ? false : p.right_transpose;
  if (module::isUniformQuantized(getInput())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = p.right_zp;
    spec.izp_const_val = p.input_zp;
    if (module::isUniformQuantized(getOutput())) {
      auto rshift_v = module::getI64Array(getRshifts());
      auto multiplier_v = module::getI64Array(getMultipliers());
      spec.requant_mode = static_cast<int>(getQuantMode());
      spec.mul_val = multiplier_v->at(0);
      spec.shift_val = -rshift_v->at(0);
      auto output_type = module::getUniformQuantizedType(getOutput());
      spec.offset_val = output_type.getZeroPoint();
      spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
      spec.fuse_rq = getFuseRq();
      if (spec.fuse_rq)
        spec.round_mode = (RoundingMode)getRoundMode();
    }
  }

  // Need to check if multi_core attribute has been set cuz we will call
  // codegen interface at layergroup stage as well
  if (supportMultiCore(*this)) {
    if (!module::isNone(getBuffer())) {
      spec.need_buffer = true;
    }

    // Double check support batch_matmul_fix8b on bm1690
    // TODO: remove this check when bm1690 support multi-core backend op
    if (module::isBM1690Family()) {
      auto in_type = module::getStorageType(getOperand(0));
      if (in_type.isInteger(8) && !p.left_transpose && p.batch == 1) {
        return BM168x::call_global_func("backend_api_fc_global", &spec,
                                        sizeof(spec), input_spec->data(),
                                        output_spec->data());
      }
    }
    return BM168x::call_global_func("backend_api_fc_multi_core_global", &spec,
                                    sizeof(spec), input_spec->data(),
                                    output_spec->data());
  }

  return BM168x::call_global_func("backend_api_fc_global", &spec, sizeof(spec),
                                  input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// ======q======================
int64_t tpu::MatMulOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto p = parseParam();
  int64_t n0, c0, h0, w0;
  module::getNCHW(getInput(), n0, c0, h0, w0, group_type);
  int64_t n1, c1, h1, w1;
  module::getNCHW(getRight(), n1, c1, h1, w1, group_type);
  auto odtype = module::getStorageType(getOutput());
  // batch dim is 1
  int64_t oshape[4] = {1, 1, 1, 1};
  if (!p.left_transpose && !p.right_transpose) {
    oshape[1] = out_cslice;
    if (p.hdim_is_batch) {
      oshape[3] = w1;
    } else {
      oshape[2] = h1;
    }
  } else if (!p.left_transpose && p.right_transpose) {
    oshape[1] = out_cslice;
    if (p.hdim_is_batch) {
      oshape[3] = c1;
    } else {
      oshape[2] = c1;
    }
  } else if (p.left_transpose) {
    oshape[1] = c1;
    if (p.hdim_is_batch) {
      oshape[3] = w0;
    } else {
      oshape[2] = h0;
    }
  }

  int64_t buffer_size = 0;
  // loop for Y_row to decrase local memory buffer
  auto in_type = BM168x::getDataType(getInput());
  auto out_type = BM168x::getDataType(getOutput());
  int in_type_len = BM168x::getFmtBytes(in_type);
  int out_type_len = BM168x::getFmtBytes(out_type);
  if (p.hdim_is_batch && in_hslice > 1) {
    /// if use buffer optimize, L_row_slice == NPU_NUM
    int64_t mm2_cycle = (ceiling_func(p.left_transpose ? oshape[1] : w0,
                                      BM168x::eu_num(in_type_len)) *
                         ceiling_func(oshape[3], (int64_t)4));
    bool buffer_optimize = mm2_cycle > (int64_t)200;
    // arrange left
    int64_t shape[4] = {1, oshape[1], 1, w0};
    if ((w0 * in_type_len) % Arch::EU_BYTES != 0 ||
        (c0 > BM168x::NPU_NUM && (p.left_transpose || !buffer_optimize))) {
      buffer_size += in_type_len * ceiling_func(shape[1], BM168x::NPU_NUM) *
                     align_up(shape[3], BM168x::eu_num(in_type_len));
    }
    // arrange right
    shape[1] = c1;
    shape[3] = w1;
    if ((w1 * in_type_len) % Arch::EU_BYTES != 0 ||
        (c1 > BM168x::NPU_NUM && (!p.left_transpose || !buffer_optimize))) {
      buffer_size += in_type_len * ceiling_func(shape[1], BM168x::NPU_NUM) *
                     align_up(shape[3], BM168x::eu_num(in_type_len));
    }

    oshape[1] =
        buffer_optimize ? std::min(oshape[1], BM168x::NPU_NUM) : oshape[1];
    if (in_type_len == 1 && out_type_len != 4) {
      // store mm2 output as int32
      buffer_size += sizeof(int32_t) *
                     ceiling_func(oshape[1], BM168x::NPU_NUM) *
                     align_up(oshape[3], BM168x::eu_num(sizeof(int32_t)));
    } else if (oshape[1] > BM168x::NPU_NUM ||
               (((oshape[3] * out_type_len) % Arch::EU_BYTES) != 0)) {
      // store output
      if (!odtype.isFloat8E4M3FN()) {
        buffer_size += out_type_len * ceiling_func(oshape[1], BM168x::NPU_NUM) *
                       align_up(oshape[3], BM168x::eu_num(out_type_len));
      } else {
        buffer_size += sizeof(int32_t) *
                       ceiling_func(oshape[1], BM168x::NPU_NUM) *
                       align_up(oshape[3], BM168x::eu_num(sizeof(int32_t)));
      }
    }
  } else if (in_type_len == 1 && out_type_len != 4) {
    if (!odtype.isFloat8E4M3FN()) {
      if (!p.left_transpose && !p.hdim_is_batch) {
        if (n0 != n1) {
          // need broadcast, if n1 = 1, move n0 to c0; else n0 = 1, keep
          // oshape[1]
          oshape[1] *= n0;
        }
      }
      bool buffer_optimize =
          (ceiling_func(p.left_transpose ? oshape[1] : h0 * w0,
                        BM168x::eu_num(in_type_len)) *
           ceiling_func(oshape[2] * oshape[3], (int64_t)4)) > 200;
      oshape[1] =
          buffer_optimize ? std::min(oshape[1], BM168x::NPU_NUM) : oshape[1];
      // store mm2 output as int32
      buffer_size += sizeof(int32_t) *
                     ceiling_func(oshape[1], BM168x::NPU_NUM) *
                     align_up(p.hdim_is_batch ? oshape[3] : oshape[2],
                              BM168x::eu_num(sizeof(int32_t)));
    } else {
      buffer_size += sizeof(int32_t) * n0 *
                     ceiling_func(oshape[1], BM168x::NPU_NUM) *
                     align_up(p.hdim_is_batch ? oshape[3] : oshape[2],
                              BM168x::eu_num(sizeof(int32_t)));
    }
  }

  if (p.input_zp != 0) {
    buffer_size +=
        align_up(p.hdim_is_batch ? w1 : h1, BM168x::eu_num(sizeof(int32_t))) *
        sizeof(int32_t);
    if (p.right_transpose) {
      buffer_size += ceiling_func(c1, BM168x::NPU_NUM) *
                     align_up(1, BM168x::eu_num(sizeof(int32_t))) *
                     sizeof(int32_t);
    }
  }

  return buffer_size;
}

void tpu::MatMulOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto p = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type, n_step, h_step,
                                           d_step, w_step, c_step);
  auto output_spec = BM168x::get_output_spec(op, group_type, n_step, h_step,
                                             d_step, w_step, c_step);
  const auto &gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);

  batch_matmul_local_spec_t param{0};
  param.buffer_addr = gi.buffer_addr;
  auto &common = param.common;
  common.Y_dtype = output_spec->at(0).dtype;
  common.L_trans = p.left_transpose;
  common.R_trans = p.right_transpose;
  common.has_bias = p.with_bias;
  common.hdim_is_batch = p.hdim_is_batch;
  common.left_reuse = p.left_reuse;
  common.requant_mode = -1;
  common.do_relu = p.do_relu;
  common.upper_limit = p.relu_limit;
  auto odtype = module::getStorageType(getOutput());
  if (module::isUniformQuantized(getInput())) {
    common.R_zp_is_const = true;
    common.R_zp_const_val = p.right_zp;
    common.izp_const_val = p.input_zp;
    if (module::isUniformQuantized(getOutput())) {
      common.requant_mode = static_cast<int>(getQuantMode());
      auto rshift_v = module::getI64Array(getRshifts());
      auto multiplier_v = module::getI64Array(getMultipliers());
      common.mul_val = multiplier_v->at(0);
      common.shift_val = -rshift_v->at(0);
      auto output_type = module::getUniformQuantizedType(getOutput());
      common.offset_val = output_type.getZeroPoint();
      common.fuse_rq = getFuseRq();
      if (common.fuse_rq)
        common.round_mode = (RoundingMode)getRoundMode();
    }
  } else if (odtype.isFloat8E4M3FN()) {
    common.requant_mode = 0;
    f64_array_t scales = module::getF64Array(getOutF8Scales().value());
    float scale = scales->at(0);
    common.mul_val = *(int *)&scale;
  }

  if (module::isDebugCmdEnable("codegen_debug")) {
    llvm::errs() << "n_step:" << n_step << ", c_step:" << c_step
                 << ", h_step:" << h_step << "\n";
    sec_info.print();
  }

  BM168x::call_local_func("backend_api_batch_matmul_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::MatMulOp::dyn_codegen_global_bm1684x(void *buffer) {
  auto p = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  if (!buffer)
    return (DYN_BMM_CONDITION(p, input_spec))
               ? sizeof(batch_matmul_common_spec_t)
               : sizeof(fc_global_spec_t);
  if (DYN_BMM_CONDITION(p, input_spec)) {
    batch_matmul_common_spec_t spec{0};
    spec.Y_dtype = output_spec->at(0).dtype;
    spec.L_trans = p.left_transpose;
    spec.R_trans = p.right_transpose;
    spec.has_bias = p.with_bias;
    spec.hdim_is_batch = p.hdim_is_batch;
    spec.requant_mode = -1;
    spec.do_relu = p.do_relu;
    spec.upper_limit = p.relu_limit;
    auto odtype = module::getStorageType(getOutput());
    if (module::isUniformQuantized(getInput())) {
      spec.R_zp_is_const = true;
      spec.R_zp_const_val = p.right_zp;
      spec.izp_const_val = p.input_zp;
      if (module::isUniformQuantized(getOutput())) {
        spec.requant_mode = static_cast<int>(getQuantMode());
        auto rshift_v = module::getI64Array(getRshifts());
        auto multiplier_v = module::getI64Array(getMultipliers());
        spec.mul_val = multiplier_v->at(0);
        spec.shift_val = -rshift_v->at(0);
        auto output_type = module::getUniformQuantizedType(getOutput());
        spec.offset_val = output_type.getZeroPoint();
        spec.fuse_rq = getFuseRq();
        spec.round_mode = (RoundingMode)getRoundMode();
      }
    } else if (odtype.isFloat8E4M3FN()) {
      spec.requant_mode = 0;
      f64_array_t scales = module::getF64Array(getOutF8Scales().value());
      float scale = scales->at(0);
      spec.mul_val = *(int *)&scale;
    }
    return BM168x::dynamic_spec_to_buffer(buffer, spec);
  }
  fc_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  spec.if_relu = p.do_relu;
  spec.relu_limit = p.relu_limit;
  spec.have_bias = p.with_bias;
  spec.requant_mode = -1;
  spec.R_transpose = p.right_transpose;
  if (module::isUniformQuantized(getInput())) {
    spec.rshift = 0;
    spec.is_asymmetric = 1;
    spec.rzp_is_const = 1;
    spec.rzp_const_val = p.right_zp;
    spec.izp_const_val = p.input_zp;
    if (module::isUniformQuantized(getOutput())) {
      auto rshift_v = module::getI64Array(getRshifts());
      auto multiplier_v = module::getI64Array(getMultipliers());
      spec.requant_mode = static_cast<int>(getQuantMode());
      spec.mul_val = multiplier_v->at(0);
      spec.shift_val = -rshift_v->at(0);
      auto output_type = module::getUniformQuantizedType(getOutput());
      spec.offset_val = output_type.getZeroPoint();
      spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
      spec.fuse_rq = getFuseRq();
      if (spec.fuse_rq)
        spec.round_mode = (RoundingMode)getRoundMode();
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::MatMulOp::get_fw_type_bm1684x() {
  auto op = getOperation();
  auto p = parseParam();
  auto input_spec = BM168x::get_input_spec(op);
  return (DYN_BMM_CONDITION(p, input_spec) ? FW_BMNET_BATCH_MATMUL
                                           : FW_BMNET_FC);
}

int64_t tpu::MatMulOp::dyn_codegen_local_bm1684x(void *buffer) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
