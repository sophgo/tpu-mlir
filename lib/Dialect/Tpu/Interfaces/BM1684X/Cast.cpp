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
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t requant_addr;
  uint32_t buffer_local_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  float offset_value;
  int input_dtype;
  int output_dtype;
  int mode;
} requant_fp_param_t;

typedef struct {
  uint64_t input_addr;
  uint64_t output_addr;
  uint64_t dequant_addr;
  int n;
  int c;
  int h;
  int w;
  bool is_perchannel;
  float scale_value;
  int offset_value;
  int input_dtype;
  int output_dtype;
} dequant_fp_param_t;

typedef struct cast_common_spec {
  int src_dtype;
  int dst_dtype;
  int round_mode;
} cast_common_spec_t;

typedef struct cast_global_spec {
  cast_common_spec_t common;
} cast_global_spec_t;

typedef struct cast_local_spec {
  cast_common_spec_t common;
  uint32_t buffer_addr;
} cast_local_spec_t;

typedef struct cast_local_param {
  cast_local_spec_t spec;
} cast_local_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================

// int8
void tpu::CastOp::codegen_global_bm1684x() {
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  auto op = getOperation();
  if (!qInput && !qOutput) {
    cast_global_spec_t spec = {0};
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;

    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    BM168x::call_global_func("backend_api_cast_global", &spec, sizeof(spec),
                             input_spec->data(), output_spec->data());

  } else {
    if (!qInput && qOutput) {
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
      BM168x::call_global_func("backend_api_requant_float_global", &param,
                               sizeof(param));
    } else if (qInput && !qOutput) {
      auto qtype = module::getUniformQuantizedType(getInput());
      dequant_fp_param_t param = {0};
      param.input_addr = module::getAddress(getInput());
      param.output_addr = module::getAddress(getOutput());
      param.n = (int)n;
      param.c = (int)c;
      param.h = (int)h;
      param.w = (int)w;
      param.is_perchannel = false;
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      BM168x::call_global_func("backend_api_dequant_float_global", &param,
                               sizeof(param));
    }
  }
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::CastOp::getBufferSize_bm1684x(int64_t in_lmem_bytes,
                                           int64_t out_lmem_bytes,
                                           int64_t in_nslice, int64_t in_hslice,
                                           int64_t out_nslice,
                                           int64_t out_hslice) {
  if (getInput().hasOneUse()) {
    return 0;
  }
  if (module::isUniformQuantized(getInput())) {
    return 0;
  }
  return in_lmem_bytes;
}

void tpu::CastOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                        local_sec_info_t &sec_info) {
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto op = getOperation();
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  auto gi = getGroupInfo(n_step, h_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);

  if (!qInput && !qOutput) {
    cast_local_spec_t spec = {0};
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;

    auto input_spec = BM168x::get_input_spec(op);
    auto output_spec = BM168x::get_output_spec(op);
    BM168x::call_local_func("backend_api_cast_local", &spec, sizeof(spec),
                            &sec_info, input_spec->data(), output_spec->data());
  } else {
    if (!qInput && qOutput) {
      auto qtype = module::getUniformQuantizedType(getOutput());
      uint32_t buffer_addr =
          getInput().hasOneUse() ? in_gi.out_addr : gi.buffer_addr;
      requant_fp_param_t param = {0};
      param.input_addr = in_gi.out_addr;
      param.output_addr = gi.out_addr;
      param.requant_addr = 0;
      param.buffer_local_addr = buffer_addr;
      param.n = sec_info.out_n_slice;
      param.c = c;
      param.h = sec_info.out_h_slice;
      param.w = w;
      param.is_perchannel = false;
      param.scale_value = 1 / qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      param.mode = ROUND_INF;
      BM168x::call_local_func("backend_api_requant_float_local", &param,
                              sizeof(param));
    } else {
      auto qtype = module::getUniformQuantizedType(getInput());
      dequant_fp_param_t param = {0};
      param.input_addr = in_gi.out_addr;
      param.output_addr = gi.out_addr;
      param.dequant_addr = 0;
      param.n = sec_info.out_n_slice;
      param.c = c;
      param.h = sec_info.out_h_slice;
      param.w = w;
      param.is_perchannel = false;
      param.scale_value = qtype.getScale();
      param.offset_value = qtype.getZeroPoint();
      param.input_dtype = BM168x::getDataType(getInput());
      param.output_dtype = BM168x::getDataType(getOutput());
      BM168x::call_local_func("backend_api_dequant_float_local", &param,
                              sizeof(param));
    }
  }
}

//dynamic codegen
int64_t tpu::CastOp::dyn_codegen_local_bm1684x(void *buffer) {
  if (!buffer) return sizeof(cast_local_spec_t);
  cast_local_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  if (!qInput && !qOutput) {
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;
    spec.buffer_addr = -1;
  } else {
    llvm_unreachable("Not Implemented");
  }
  auto p = static_cast<char *>(buffer);
  memcpy(p, &spec, sizeof(spec));
  p += sizeof(spec);
  return p - static_cast<char *>(buffer);
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::CastOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer) return sizeof(cast_global_spec_t);
  cast_global_spec_t spec;
  memset(&spec, 0, sizeof(spec));
  bool qInput = module::isUniformQuantized(getInput());
  bool qOutput = module::isUniformQuantized(getOutput());
  if (!qInput && !qOutput) {
    spec.common.src_dtype = BM168x::getDataType(getInput());
    spec.common.dst_dtype = BM168x::getDataType(getOutput());
    spec.common.round_mode = ROUND_INF;
  } else {
    llvm_unreachable("Not Implemented");
  }
  auto p = static_cast<char *>(buffer);
  memcpy(p, &spec, sizeof(spec));
  p += sizeof(spec);
  return p - static_cast<char *>(buffer);
}
