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

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::CastOp::codegen_global_bm1684x() {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto round_mode = round_mode_convert(getRoundMode());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto op = getOperation();
  if (!(fInput && qOutput) && !(qInput && fOutput)) {
    cast_global_spec_t spec = {0};
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = round_mode;

    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    BM168x::call_global_func("backend_api_cast_global", &spec, sizeof(spec),
                             input_spec->data(), output_spec->data());

  } else {
    if (fInput && qOutput) {
      auto qtype = module::getUniformQuantizedType(getOutput());
      requant_fp_param_t param = {0};
      param.input_addr = module::getAddress(getInput());
      param.output_addr = module::getAddress(getOutput());
      param.n = (int)n;
      param.c = (int)c;
      param.h = (int)h;
      param.w = (int)w;
      param.is_perchannel = false;
      param.scale_value = 1.0 / qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.mode = 0;
      param.round_mode = round_mode;
      BM168x::call_global_func("backend_api_requant_float_global", &param,
                               sizeof(param));
    } else if (qInput && fOutput) {
      auto qtype = module::getUniformQuantizedType(getInput());
      dequant_fp_param_t param = {0};
      param.input_addr = module::getAddress(getInput());
      param.output_addr = module::getAddress(getOutput());
      param.n = (int)n;
      param.c = (int)c;
      param.h = (int)h;
      param.w = (int)w;

      auto value = getInput();
      auto stype = module::getStorageType(value);
      if (stype.isInteger(4)) {
        auto op = value.getDefiningOp();
        if (isa<tpu::MatMulOp>(op)) {
          auto p = dyn_cast<tpu::MatMulOp>(op).parseParam();
          param.n = p.batch;
          param.c = p.M;
          param.h = p.N;
          param.w = 1;
        }
      }
      param.is_perchannel = false;
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.round_mode = round_mode;
      BM168x::call_global_func("backend_api_dequant_float_global", &param,
                               sizeof(param));
    }
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CastOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  // if (getInput().hasOneUse()) {
  //   return 0;
  // }
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  if (!(qInput && fOutput) && !(fInput && qOutput)) {
    return 0;
  } else {
    if (fInput && qOutput) {
      if (getInput().hasOneUse()) {
        return 0;
      }
      return in_lmem_bytes;
    } else {
      if (out_type.isF16() || out_type.isBF16()) {
        return out_lmem_bytes;
      }
      return 0;
    }
  }
}

void tpu::CastOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                        int64_t h_step, int64_t d_step,
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  int64_t n, c, d, h, w;
  module::getNCDHW(getInput(), n, c, d, h, w, group_type);
  auto op = getOperation();
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);

  if (!(qInput && fOutput) && !(fInput && qOutput)) {
    cast_local_spec_t spec = {0};
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;

    auto input_spec = BM168x::get_input_spec(op, group_type, n_step, h_step,
                                             d_step, w_step, c_step);
    auto output_spec = BM168x::get_output_spec(op, group_type, n_step, h_step,
                                               d_step, w_step, c_step);
    BM168x::call_local_func("backend_api_cast_local", &spec, sizeof(spec),
                            &sec_info, input_spec->data(), output_spec->data());
  } else {
    if (fInput && qOutput) {
      auto qtype = module::getUniformQuantizedType(getOutput());
      uint32_t buffer_addr =
          getInput().hasOneUse() ? in_gi.out_addr : gi.buffer_addr;
      requant_fp_param_t param = {0};
      param.input_addr = in_gi.out_addr;
      param.output_addr = gi.out_addr;
      param.requant_addr = 0;
      param.buffer_local_addr = buffer_addr;
      param.n = sec_info.out_n_slice * in_gi.d_slice;
      param.c = sec_info.c_slice;
      param.h = sec_info.out_h_slice;
      param.w = sec_info.out_w_slice;
      param.is_perchannel = false;
      param.scale_value = 1 / qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.mode = 0;
      param.round_mode = ROUND_INF;
      BM168x::call_local_func("backend_api_requant_float_local", &param,
                              sizeof(param));
    } else {
      auto qtype = module::getUniformQuantizedType(getInput());
      dequant_fp_param_t param = {0};
      param.input_addr = in_gi.out_addr;
      param.output_addr = gi.out_addr;
      param.dequant_addr = 0;
      param.buffer_addr = gi.buffer_addr;
      param.n = sec_info.out_n_slice * in_gi.d_slice;
      param.c = sec_info.c_slice;
      param.h = sec_info.out_h_slice;
      param.w = sec_info.out_w_slice;
      param.is_perchannel = false;
      param.has_buffer = (out_type.isF16() || out_type.isBF16());
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.margins = sec_info.hw_margins_opdA;
      BM168x::call_local_func("backend_api_dequant_float_local", &param,
                              sizeof(param));
    }
  }
}

// dynamic codegen
int64_t tpu::CastOp::dyn_codegen_local_bm1684x(void *buffer) {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  if (!(fInput && qOutput) && !(qInput && fOutput)) {
    if (!buffer)
      return sizeof(cast_local_param_t);
    cast_local_param_t param;
    memset(&param, 0, sizeof(param));
    param.spec.common.src_dtype = BM168x::getDataType(getInput());
    param.spec.common.dst_dtype = BM168x::getDataType(getOutput());
    param.spec.common.round_mode = ROUND_INF;
    param.spec.buffer_addr = -1;
    return BM168x::dynamic_spec_to_buffer(buffer, param);
  } else {
    if (fInput && qOutput) {
      if (!buffer)
        return sizeof(dyn_requant_fp_local_param_t);
      auto qtype = module::getUniformQuantizedType(getOutput());
      uint32_t buffer_addr =
          getInput().hasOneUse() ? in_gi.out_addr : gi.buffer_addr;
      dyn_requant_fp_local_param_t param = {0};
      param.buffer_local_addr = buffer_addr;
      param.common.is_perchannel = false;
      param.common.scale_value = 1 / qtype.getScale();
      param.common.offset_value = qtype.getZeroPoint();
      param.common.output_dtype = BM168x::getDataType(getOutput());
      param.common.mode = ROUND_INF; // ToDo: need further check
      param.common.round_mode = ROUND_INF;
      return BM168x::dynamic_spec_to_buffer(buffer, param);
    } else {
      if (!buffer)
        return sizeof(dyn_dequant_fp_param_t);
      auto qtype = module::getUniformQuantizedType(getInput());
      dyn_dequant_fp_param_t param = {0};
      param.is_perchannel = false;
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.output_dtype = BM168x::getDataType(getOutput());
      return BM168x::dynamic_spec_to_buffer(buffer, param);
    }
  }
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CastOp::dyn_codegen_global_bm1684x(void *buffer) {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  if (!(fInput && qOutput) && !(qInput && fOutput)) {
    if (!buffer)
      return sizeof(cast_global_spec_t);
    cast_global_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;
    return BM168x::dynamic_spec_to_buffer(buffer, spec);
  } else {
    if (fInput && qOutput) {
      if (!buffer)
        return sizeof(dyn_requant_fp_global_param_t);
      auto qtype = module::getUniformQuantizedType(getOutput());
      dyn_requant_fp_global_param_t param = {0};
      param.common.is_perchannel = false;
      param.common.scale_value = 1.0 / qtype.getScale();
      param.common.offset_value = qtype.getZeroPoint();
      param.common.output_dtype = BM168x::getDataType(getOutput());
      param.common.mode = 0; // ToDo: need further check
      param.common.round_mode = ROUND_INF;
      return BM168x::dynamic_spec_to_buffer(buffer, param);
    } else {
      if (!buffer)
        return sizeof(dyn_dequant_fp_param_t);
      auto qtype = module::getUniformQuantizedType(getInput());
      dyn_dequant_fp_param_t param = {0};
      param.is_perchannel = false;
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.output_dtype = BM168x::getDataType(getOutput());
      return BM168x::dynamic_spec_to_buffer(buffer, param);
    }
  }
}

int64_t tpu::CastOp::get_fw_type_bm1684x() {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  bool fInput = in_type.isIntOrIndex() == false;
  bool fOutput = out_type.isIntOrIndex() == false;
  if (!(fInput && qOutput) && !(qInput && fOutput))
    return FW_BMNET_DTYPE_CONVERT;
  else {
    if (fInput && qOutput)
      return FW_BMNET_REQUANT_FP32;
    else
      return FW_BMNET_DEQUANT_FP32;
  }
}
