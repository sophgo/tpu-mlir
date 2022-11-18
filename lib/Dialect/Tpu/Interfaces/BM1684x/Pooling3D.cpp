//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pooling3d_spec {
  int64_t input_addr;
  int64_t output_addr;
  int32_t buffer_addr;
  int32_t input_shape[5];
  int32_t output_shape[5];
  int32_t *kernel;
  int32_t *stride;
  int32_t *dilation;
  int32_t *pad;
  bool is_avg_pooling;
  int32_t avg_pooling_mode;
  int32_t avg_rd_mode;
  /* for float */
  int32_t if_relu;
  float relu_limit;
  int32_t in_dtype;
  int32_t out_dtype;
  /* for fix8b */
  int32_t avg_pooling_quant_mode;
  bool merge_requant;
  float rq_scale;
  float rq_offset;
} pooling3d_spec_t;

#ifdef __cplusplus
}
#endif

static bool has_pad(const pool_attr_t &attrs) {
  if (attrs.pad_h != 0 || attrs.pad_w != 0 || attrs.pad_d != 0)
    return true;
  if ((attrs.ih - attrs.kh) % attrs.sh != 0 ||
      (attrs.iw - attrs.kw) % attrs.sw != 0 ||
      (attrs.id - attrs.kd) % attrs.sd != 0)
    return true;
  if ((attrs.ih - attrs.kh) / attrs.sh + 1 != attrs.oh ||
      (attrs.iw - attrs.kw) / attrs.sw + 1 != attrs.ow ||
      (attrs.id - attrs.kd) / attrs.sd + 1 != attrs.od)
    return true;
  return false;
}

// =========================================
// GlobalGenInterface
// =========================================

void tpu::Pool3DOp::codegen_global_bm1684x() {
  pool_attr_t attrs;
  parseParam(&attrs);

  auto op = getOperation();
  pooling3d_spec_t spec = {0};
  spec.input_addr = Module::getAddress(input());
  spec.output_addr = Module::getAddress(output());
  spec.buffer_addr = -1;
  spec.input_shape[0] = attrs.n;
  spec.input_shape[1] = attrs.c;
  spec.input_shape[2] = attrs.id;
  spec.input_shape[3] = attrs.ih;
  spec.input_shape[4] = attrs.iw;
  spec.output_shape[0] = attrs.n;
  spec.output_shape[1] = attrs.c;
  spec.output_shape[2] = attrs.od;
  spec.output_shape[3] = attrs.oh;
  spec.output_shape[4] = attrs.ow;
  spec.in_dtype = BM168x::getDataType(input());
  spec.out_dtype = BM168x::getDataType(output());

  int32_t kernel[3] = {(int32_t)attrs.kd, (int32_t)attrs.kh, (int32_t)attrs.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attrs.sd, (int32_t)attrs.sh,
                        (int32_t)attrs.sw};
  int32_t pads[6] = {(int32_t)attrs.pad_d, (int32_t)attrs.pad_d_after,
                     (int32_t)attrs.pad_h, (int32_t)attrs.pad_h_after,
                     (int32_t)attrs.pad_w, (int32_t)attrs.pad_w_after};
  spec.kernel = kernel;
  spec.dilation = dilation;
  spec.stride = strides;
  spec.pad = pads;
  spec.is_avg_pooling = false;
  spec.avg_pooling_mode = attrs.count_include_pad ? 0 : 1;
  spec.avg_rd_mode = ROUND_UP;
  spec.if_relu = attrs.do_relu;
  spec.relu_limit = attrs.relu_limit;

  if (pool_mode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (Quant::isUniformQuantized(input())) {
      bool with_pad = has_pad(attrs) && attrs.count_include_pad == 0;
      spec.avg_pooling_quant_mode = with_pad ? 1 : 2;
      // if (spec.avg_pooling_quant_mode == 0) {
      //   spec.multiplier = multiplier().has_value() ? multiplier().value() :
      //   1; spec.rshiftbits = rshift().has_value() ? rshift().value() : 0;
      // }
      if (spec.avg_pooling_quant_mode == 2) {
        spec.merge_requant = true;
        spec.rq_scale =
            scale().has_value() ? (scale().value().convertToDouble()) : 1.;
        spec.rq_offset =
            offset().has_value() ? (offset().value().convertToDouble()) : 0.;
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
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  auto module = Module::getModuleOp(getOperation());
  auto out_dtype = output().getType();
  pool_attr_t attrs;
  parseParam(&attrs);

  auto op = getOperation();
  auto *instance = BM168x::inst;
  int c_per_npu = ceiling_func(attrs.c, BM168x::NPU_NUM);

  if (attrs.kd > 1 || attrs.sd > 1 || attrs.pad_d > 0 ||
      attrs.pad_d_after > 0) {
    /// pooling include depth-dimention
    if (out_dtype.isInteger(8) && pool_mode() == tpu::PoolMode::Avg) {
      int64_t dtype_bytes =
          attrs.kd * attrs.kh * attrs.kw > 256 ? sizeof(int) : sizeof(short);
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      buffer_size = (1 + attrs.od) * align_up(out_hslice * attrs.ow, eu_num) *
                    c_per_npu * dtype_bytes;
    } else {
      int64_t dtype_bytes = BM168x::getFmtBytes(BM168x::getDataType(output()));
      int64_t eu_num = BM168x::eu_num(dtype_bytes);
      buffer_size =
          align_up(out_hslice * attrs.ow, eu_num) * c_per_npu * dtype_bytes;
    }
  } else if (out_dtype.isInteger(8) && pool_mode() == tpu::PoolMode::Avg) {
    int64_t dtype_bytes = attrs.kd * attrs.kh * attrs.kw > 256
                              ? sizeof(int32_t)
                              : sizeof(int16_t);
    int64_t eu_num = BM168x::eu_num(dtype_bytes);
    buffer_size = align_up(out_hslice * attrs.ow, eu_num) *
                  ceiling_func(attrs.c, BM168x::NPU_NUM) * dtype_bytes;
  }
  return buffer_size;
}

void tpu::Pool3DOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step) {
  pool_attr_t attrs;
  parseParam(&attrs);
  auto op = getOperation();
  auto module = Module::getModuleOp(op);
  auto in_gi = LocalGenInterface::getGroupInfo(input(), n_step, h_step);
  auto gi = getGroupInfo(n_step, h_step);
  pooling3d_spec_t spec = {0};
  spec.input_addr = in_gi.out_addr;
  spec.output_addr = gi.out_addr;
  spec.buffer_addr = gi.buffer_addr;
  spec.input_shape[0] = in_gi.n_slice;
  spec.input_shape[1] = attrs.c;
  spec.input_shape[2] = attrs.id;
  spec.input_shape[3] = in_gi.h_slice;
  spec.input_shape[4] = attrs.iw;
  spec.output_shape[0] = gi.n_slice;
  spec.output_shape[1] = attrs.c;
  spec.output_shape[2] = attrs.od;
  spec.output_shape[3] = gi.h_slice;
  spec.output_shape[4] = attrs.ow;
  spec.in_dtype = BM168x::getDataType(input());
  spec.out_dtype = BM168x::getDataType(output());

  int32_t kernel[3] = {(int32_t)attrs.kd, (int32_t)attrs.kh, (int32_t)attrs.kw};
  int32_t dilation[3] = {1, 1, 1};
  int32_t strides[3] = {(int32_t)attrs.sd, (int32_t)attrs.sh,
                        (int32_t)attrs.sw};
  int32_t pads[6] = {(int32_t)attrs.pad_d, (int32_t)attrs.pad_d_after,
                     (int32_t)attrs.pad_h, (int32_t)attrs.pad_h_after,
                     (int32_t)attrs.pad_w, (int32_t)attrs.pad_w_after};
  pads[2] = (in_gi.h_idx == 0 ? attrs.pad_h : 0);
  pads[3] = (in_gi.h_idx + in_gi.h_slice == attrs.ih ? attrs.pad_h_after : 0);
  spec.kernel = kernel;
  spec.dilation = dilation;
  spec.stride = strides;
  spec.pad = pads;

  spec.avg_pooling_mode = attrs.count_include_pad ? 0 : 1;
  spec.avg_rd_mode = ROUND_UP;
  spec.is_avg_pooling = false;

  if (pool_mode() == tpu::PoolMode::Avg) {
    spec.is_avg_pooling = true;
    if (Quant::isUniformQuantized(input())) {
      bool with_pad = has_pad(attrs) && attrs.count_include_pad == 0;
      spec.avg_pooling_quant_mode = with_pad ? 1 : 2;

      // if (spec.avg_pooling_quant_mode == 0) {
      //   spec.multiplier = multiplier().has_value() ? multiplier().value() :
      //   1; spec.rshiftbits = rshift().has_value() ? rshift().value() : 0;
      // }
      if (spec.avg_pooling_quant_mode == 2) {
        spec.merge_requant = true;
        spec.rq_scale =
            scale().has_value() ? (scale().value().convertToDouble()) : 1.;
        spec.rq_offset =
            offset().has_value() ? (offset().value().convertToDouble()) : 0.;
      }
    }
  }

  spec.if_relu = attrs.do_relu;
  spec.relu_limit = attrs.relu_limit;

  BM168x::call_local_func("backend_api_pool3d_local", &spec,
                          sizeof(pooling3d_spec_t));
}
