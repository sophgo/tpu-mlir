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

static bool has_pad(const pool_attr_t &attr) {
  if (attr.pad_h != 0 || attr.pad_w != 0 || attr.pad_d != 0)
    return true;
  if ((attr.ih - attr.kh) % attr.sh != 0 ||
      (attr.iw - attr.kw) % attr.sw != 0 || (attr.id - attr.kd) % attr.sd != 0)
    return true;
  if ((attr.ih - attr.kh) / attr.sh + 1 != attr.oh ||
      (attr.iw - attr.kw) / attr.sw + 1 != attr.ow ||
      (attr.id - attr.kd) / attr.sd + 1 != attr.od)
    return true;
  return false;
}

// =========================================
// GlobalGenInterface
// =========================================

void tpu::Pool3DOp::codegen_global_bm1684x() {
  auto attr = parseParam();

  pooling3d_spec_t spec = {0};
  spec.input_addr = module::getAddress(getInput());
  spec.output_addr = module::getAddress(getOutput());
  spec.buffer_addr = -1;
  spec.input_shape[0] = attr.n;
  spec.input_shape[1] = attr.c;
  spec.input_shape[2] = attr.id;
  spec.input_shape[3] = attr.ih;
  spec.input_shape[4] = attr.iw;
  spec.output_shape[0] = attr.n;
  spec.output_shape[1] = attr.c;
  spec.output_shape[2] = attr.od;
  spec.output_shape[3] = attr.oh;
  spec.output_shape[4] = attr.ow;
  spec.in_dtype = BM168x::getDataType(getInput());
  spec.out_dtype = BM168x::getDataType(getOutput());

  int32_t kernel[3] = {(int32_t)attr.kd, (int32_t)attr.kh, (int32_t)attr.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attr.sd, (int32_t)attr.sh, (int32_t)attr.sw};
  int32_t pads[6] = {(int32_t)attr.pad_d, (int32_t)attr.pad_d_after,
                     (int32_t)attr.pad_h, (int32_t)attr.pad_h_after,
                     (int32_t)attr.pad_w, (int32_t)attr.pad_w_after};
  spec.kernel = kernel;
  spec.dilation = dilation;
  spec.stride = strides;
  spec.pad = pads;
  spec.is_avg_pooling = false;
  spec.avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  spec.avg_rd_mode = round_mode_convert(getRoundMode());
  spec.avg_src_rd_mode = round_mode_convert(getFirstRoundMode());
  spec.if_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
      spec.avg_pooling_quant_mode = with_pad ? 1 : 2;
      // if (spec.avg_pooling_quant_mode == 0) {
      //   spec.multiplier = getMultiplier().has_value() ?
      //   getMultiplier().value() : 1; spec.rshiftbits =
      //   getRshift().has_value() ? getRshift().value() : 0;
      // }
      if (spec.avg_pooling_quant_mode == 2) {
        spec.merge_requant = true;
        spec.rq_scale = getScale().has_value()
                            ? (getScale().value().convertToDouble())
                            : 1.;
        spec.rq_offset = getOffset().has_value()
                             ? (getOffset().value().convertToDouble())
                             : 0.;
      }
    }
  }

  BM168x::call_global_func("backend_api_pool3d_global", &spec,
                           sizeof(pooling3d_spec_t));
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::Pool3DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  int64_t buffer_size = 0;
  auto out_dtype = module::getStorageType(getOutput());
  auto attr = parseParam();

  int c_per_npu = ceiling_func(attr.c, BM168x::NPU_NUM);

  if (attr.kd > 1 || attr.sd > 1 || attr.pad_d > 0 || attr.pad_d_after > 0) {
    /// pooling include depth-dimention
    if (out_dtype.isInteger(8) && getPoolMode() == tpu::PoolMode::Avg) {
      int64_t dtype_bytes =
          attr.kd * attr.kh * attr.kw > 256 ? sizeof(int) : sizeof(short);
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      buffer_size = (1 + attr.od) * align_up(out_hslice * out_wslice, eu_num) *
                    c_per_npu * dtype_bytes;
    } else {
      auto dtype_bytes = BM168x::getFmtBytes(BM168x::getDataType(getOutput()));
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      buffer_size =
          align_up(out_hslice * out_wslice, eu_num) * c_per_npu * dtype_bytes;
    }
  } else if (out_dtype.isInteger(8) && getPoolMode() == tpu::PoolMode::Avg) {
    int64_t dtype_bytes =
        attr.kd * attr.kh * attr.kw > 256 ? sizeof(int32_t) : sizeof(int16_t);
    int64_t eu_num = BM168x::eu_num(dtype_bytes);
    buffer_size = align_up(out_hslice * out_wslice, eu_num) *
                  ceiling_func(attr.c, BM168x::NPU_NUM) * dtype_bytes;
  }
  return buffer_size;
}

void tpu::Pool3DOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                          int64_t h_step, int64_t d_step,
                                          int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  // auto op = getOperation();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);

  auto attr = parseParam();
  pooling3d_spec_t spec = {0};
  spec.input_addr = in_gi.out_addr;
  spec.output_addr = gi.out_addr;
  spec.buffer_addr = gi.buffer_addr;
  spec.input_shape[0] = sec_info.n_slice;
  spec.input_shape[1] = attr.c;
  spec.input_shape[2] = sec_info.d_slice;
  spec.input_shape[3] = sec_info.h_slice;
  spec.input_shape[4] = sec_info.w_slice;
  spec.output_shape[0] = sec_info.out_n_slice;
  spec.output_shape[1] = attr.c;
  spec.output_shape[2] =
      gi.d_slice; // sec_info.out_d_slice; // <- no such attr; need to change
                  // along with backend api_common.h:sec_info, otherwise memcpy
                  // will mess up
  spec.output_shape[3] = sec_info.out_h_slice;
  spec.output_shape[4] = sec_info.out_w_slice;
  spec.in_dtype = BM168x::getDataType(getInput());
  spec.out_dtype = BM168x::getDataType(getOutput());

  int32_t kernel[3] = {(int32_t)attr.kd, (int32_t)attr.kh, (int32_t)attr.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attr.sd, (int32_t)attr.sh, (int32_t)attr.sw};
  int32_t pads[6] = {0, 0, 0, 0, 0, 0};
  // int32_t pads[6] = {(int32_t)attr.pad_d, (int32_t)attr.pad_d_after,
  //                    (int32_t)attr.pad_h, (int32_t)attr.pad_h_after,
  //                    (int32_t)attr.pad_w, (int32_t)attr.pad_w_after};
  pads[0] = (in_gi.d_idx == 0 ? attr.pad_d : 0);
  pads[1] = (in_gi.d_idx + in_gi.d_slice == attr.id ? attr.pad_d_after : 0);
  pads[2] = (in_gi.h_idx == 0 ? attr.pad_h : 0);
  pads[3] = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.pad_h_after : 0);
  pads[4] = (in_gi.w_idx == 0 ? attr.pad_w : 0);
  pads[5] = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pad_w_after : 0);
  spec.kernel = kernel;
  spec.dilation = dilation;
  spec.stride = strides;
  spec.pad = pads;

  spec.avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  spec.avg_rd_mode = round_mode_convert(getRoundMode());
  spec.avg_src_rd_mode = round_mode_convert(getFirstRoundMode());
  spec.is_avg_pooling = false;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
      spec.avg_pooling_quant_mode = with_pad ? 1 : 2;

      // if (spec.avg_pooling_quant_mode == 0) {
      //   spec.multiplier = getMultiplier().has_value() ?
      //   getMultiplier().value() : 1; spec.rshiftbits =
      //   getRshift().has_value() ? getRshift().value() : 0;
      // }
      if (spec.avg_pooling_quant_mode == 2) {
        spec.merge_requant = true;
        spec.rq_scale = getScale().has_value()
                            ? (getScale().value().convertToDouble())
                            : 1.;
        spec.rq_offset = getOffset().has_value()
                             ? (getOffset().value().convertToDouble())
                             : 0.;
      }
    }
  }

  spec.if_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;

  BM168x::call_local_func("backend_api_pool3d_local", &spec,
                          sizeof(pooling3d_spec_t));
}

// dynamic codegen
int64_t tpu::Pool3DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_pooling3d_local_spec_t);
  auto gi = getGroupInfo(0, 0, 0, 0, 0);
  auto attr = parseParam();
  dyn_pooling3d_local_spec_t spec = {0};
  spec.buffer_addr = gi.buffer_addr;
  spec.common.output_shape[0] = attr.n;
  spec.common.output_shape[1] = attr.c;
  spec.common.output_shape[2] = attr.od;
  spec.common.output_shape[3] = attr.oh;
  spec.common.output_shape[4] = attr.ow;
  spec.common.in_dtype = BM168x::getDataType(getInput());
  spec.common.out_dtype = BM168x::getDataType(getOutput());

  int32_t kernel[3] = {(int32_t)attr.kd, (int32_t)attr.kh, (int32_t)attr.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attr.sd, (int32_t)attr.sh, (int32_t)attr.sw};
  int32_t pads[6] = {(int32_t)attr.pad_d, (int32_t)attr.pad_d_after,
                     (int32_t)attr.pad_h, (int32_t)attr.pad_h_after,
                     (int32_t)attr.pad_w, (int32_t)attr.pad_w_after};
  memcpy(spec.common.kernel, kernel, sizeof(int32_t) * 3);
  memcpy(spec.common.dilation, dilation, sizeof(int32_t) * 3);
  memcpy(spec.common.stride, strides, sizeof(int32_t) * 3);
  memcpy(spec.common.pad, pads, sizeof(int32_t) * 6);

  spec.common.avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  spec.common.avg_rd_mode = round_mode_convert(getRoundMode());
  spec.common.avg_src_rd_mode = round_mode_convert(getFirstRoundMode());
  spec.common.is_avg_pooling = false;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.common.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
      spec.common.avg_pooling_quant_mode = with_pad ? 1 : 2;
      if (spec.common.avg_pooling_quant_mode == 2) {
        spec.common.merge_requant = true;
        spec.common.rq_scale = getScale().has_value()
                                   ? (getScale().value().convertToDouble())
                                   : 1.;
        spec.common.rq_offset = getOffset().has_value()
                                    ? (getOffset().value().convertToDouble())
                                    : 0.;
      }
    }
  }

  spec.common.if_relu = attr.do_relu;
  spec.common.relu_limit = attr.relu_limit;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Pool3DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_pooling3d_global_spec_t);
  auto attr = parseParam();
  dyn_pooling3d_global_spec_t spec = {0};
  spec.buffer_addr = -1;
  spec.common.output_shape[0] = attr.n;
  spec.common.output_shape[1] = attr.c;
  spec.common.output_shape[2] = attr.od;
  spec.common.output_shape[3] = attr.oh;
  spec.common.output_shape[4] = attr.ow;
  spec.common.in_dtype = BM168x::getDataType(getInput());
  spec.common.out_dtype = BM168x::getDataType(getOutput());

  int32_t kernel[3] = {(int32_t)attr.kd, (int32_t)attr.kh, (int32_t)attr.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attr.sd, (int32_t)attr.sh, (int32_t)attr.sw};
  int32_t pads[6] = {(int32_t)attr.pad_d, (int32_t)attr.pad_d_after,
                     (int32_t)attr.pad_h, (int32_t)attr.pad_h_after,
                     (int32_t)attr.pad_w, (int32_t)attr.pad_w_after};
  memcpy(spec.common.kernel, kernel, sizeof(int32_t) * 3);
  memcpy(spec.common.dilation, dilation, sizeof(int32_t) * 3);
  memcpy(spec.common.stride, strides, sizeof(int32_t) * 3);
  memcpy(spec.common.pad, pads, sizeof(int32_t) * 6);
  spec.common.is_avg_pooling = false;
  spec.common.avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  spec.common.avg_rd_mode = round_mode_convert(getRoundMode());
  spec.common.avg_src_rd_mode = round_mode_convert(getFirstRoundMode());
  spec.common.if_relu = attr.do_relu;
  spec.common.relu_limit = attr.relu_limit;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.common.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
      spec.common.avg_pooling_quant_mode = with_pad ? 1 : 2;
      if (spec.common.avg_pooling_quant_mode == 2) {
        spec.common.merge_requant = true;
        spec.common.rq_scale = getScale().has_value()
                                   ? (getScale().value().convertToDouble())
                                   : 1.;
        spec.common.rq_offset = getOffset().has_value()
                                    ? (getOffset().value().convertToDouble())
                                    : 0.;
      }
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::Pool3DOp::get_fw_type_bm1684x() { return FW_BMNET_POOL3D; }
