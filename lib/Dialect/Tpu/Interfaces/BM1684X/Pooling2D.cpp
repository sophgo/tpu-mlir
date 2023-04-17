//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;


static void SpecAssign(const pool_attr_t &attr, pooling_common_spec_t &spec) {
  spec.kh = attr.kh;
  spec.kw = attr.kw;
  spec.pad_h_t = attr.pad_h;
  spec.pad_h_b = attr.pad_h_after;
  spec.pad_w_l = attr.pad_w;
  spec.pad_w_r = attr.pad_w_after;
  spec.stride_h = attr.sh;
  spec.stride_w = attr.sw;
  spec.dh = 1;
  spec.dw = 1;
  spec.is_global_pooling = attr.is_global;
  spec.avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  spec.if_relu = attr.do_relu;
  spec.relu_limit = attr.relu_limit;
  spec.ceil_mode = 0;
  spec.round_mode = ROUNDING_HALF_AWAY_FROM_ZERO;
  /// TODO: may be need support pad value for pooling2D and pooling 3D
  /// spec.pad_value = attr.pad_value;
}

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

void tpu::Pool2DOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto attr = parseParam();
  pooling_common_spec_t spec = {0};
  SpecAssign(attr, spec);
  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
      spec.avg_pooling_quant_mode =
          module::isAsymmetric() ? (with_pad ? 1 : 2) : 0;

      if (spec.avg_pooling_quant_mode == 0) {
        spec.multiplier =
            getMultiplier().has_value() ? getMultiplier().value() : 1;
        spec.rshiftbits = getRshift().has_value() ? getRshift().value() : 0;
      } else if (spec.avg_pooling_quant_mode == 2) {
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
  BM168x::call_global_func("backend_api_pooling_global", &spec, sizeof(spec),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::Pool2DOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_hslice, int64_t out_dslice, int64_t out_wslice,
    group_type_t group_type) {
  switch (getPoolMode()) {
  case tpu::PoolMode::Max:
    return 0;
  case tpu::PoolMode::Avg:
    int64_t size = 0;
    auto &p = getPool2DParam(*this);
    int64_t npu_num = BM168x::NPU_NUM;
    if (module::isAsymmetric()) {
      bool with_pad = has_pad(p) && p.count_include_pad == 0;
      auto kernel = module::getI64Array(getKernelShape());
      int64_t dtype_bytes = p.kh * p.kw > 256 ? sizeof(int) : sizeof(short);
      if (with_pad) {
        dtype_bytes = sizeof(int);
      }
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      size += align_up(out_hslice * out_wslice, eu_num) * ceiling_func(p.c, npu_num) * 
              dtype_bytes;
    }
    if (p.is_global) {
      auto dtype_bytes = BM168x::getFmtBytes(BM168x::getDataType(getOutput()));
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      size += out_nslice * eu_num * ceiling_func(p.c, npu_num) * dtype_bytes;
    }
    return size;
  }
  llvm_unreachable("unimplemented Pooling.");
}

void tpu::Pool2DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                          group_type_t group_type,
                                          local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step, d_step, w_step);

  auto attr = parseParam();
  pooling_local_spec_t spec = {0};
  auto &common = spec.common;
  SpecAssign(attr, common);
  spec.buffer_addr = gi.buffer_addr;
  common.pad_h_t = (in_gi.h_idx == 0 ? attr.pad_h : 0);
  common.pad_h_b =
      (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.pad_h_after : 0);
  common.pad_w_l = (in_gi.w_idx == 0 ? attr.pad_w : 0);
  common.pad_w_r =
      (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pad_w_after : 0);

  if (getPoolMode() == tpu::PoolMode::Avg) {
    bool with_pad = has_pad(attr) && attr.count_include_pad == 0;
    common.is_avg_pooling = true;
    common.avg_pooling_quant_mode =
        module::isAsymmetric() ? (with_pad ? 1 : 2) : 0;

    if (common.avg_pooling_quant_mode == 0) {
      common.multiplier =
          getMultiplier().has_value() ? getMultiplier().value() : -1;
      common.rshiftbits = getRshift().has_value() ? getRshift().value() : -1;
    } else if (common.avg_pooling_quant_mode == 2) {
      common.merge_requant = true;
      common.rq_scale =
          getScale().has_value() ? (getScale().value().convertToDouble()) : -1.;
      common.rq_offset = getOffset().has_value()
                             ? (getOffset().value().convertToDouble())
                             : -1.;
    }
  }

  BM168x::call_local_func("backend_api_pooling_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// dynamic codegen
int64_t tpu::Pool2DOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pooling_local_spec_t);
  pooling_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  auto attrs = parseParam();
  auto &common = spec.common;
  SpecAssign(attrs, common);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), 0, 0);
  auto gi = getGroupInfo(0, 0, 0, 0);

  spec.buffer_addr = gi.buffer_addr;
  common.pad_h_t = attrs.pad_h;
  common.pad_h_b = attrs.pad_h_after;

  if (getPoolMode() == tpu::PoolMode::Avg) {
    bool with_pad = has_pad(attrs) && attrs.count_include_pad == 0;
    common.is_avg_pooling = true;
    common.avg_pooling_quant_mode =
        module::isAsymmetric() ? (with_pad ? 1 : 2) : 0;

    if (common.avg_pooling_quant_mode == 0) {
      common.multiplier =
          getMultiplier().has_value() ? getMultiplier().value() : -1;
      common.rshiftbits = getRshift().has_value() ? getRshift().value() : -1;
    } else if (common.avg_pooling_quant_mode == 2) {
      common.merge_requant = true;
      common.rq_scale =
          getScale().has_value() ? (getScale().value().convertToDouble()) : -1.;
      common.rq_offset = getOffset().has_value()
                             ? (getOffset().value().convertToDouble())
                             : -1.;
    }
  }

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::Pool2DOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pooling_common_spec_t);
  pooling_common_spec_t spec;
  memset(&spec, 0, sizeof(spec));

  auto attrs = parseParam();
  SpecAssign(attrs, spec);
  if (getPoolMode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (module::isUniformQuantized(getInput())) {
      bool with_pad = has_pad(attrs) && attrs.count_include_pad == 0;
      spec.avg_pooling_quant_mode =
          module::isAsymmetric() ? (with_pad ? 1 : 2) : 0;

      if (spec.avg_pooling_quant_mode == 0) {
        spec.multiplier =
            getMultiplier().has_value() ? getMultiplier().value() : 1;
        spec.rshiftbits = getRshift().has_value() ? getRshift().value() : 0;
      } else if (spec.avg_pooling_quant_mode == 2) {
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

  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::Pool2DOp::get_fw_type_bm1684x() {
  return FW_BMNET_POOL;
}
