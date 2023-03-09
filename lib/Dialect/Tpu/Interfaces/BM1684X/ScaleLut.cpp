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
#include "tpu_mlir/Dialect/Tpu/Transforms/DynCompileCommon.hpp"
// =========================================
// GlobalGenInterface
// =========================================

using namespace tpu_mlir::backend;

void tpu::ScaleLutOp::codegen_global_bm1684x() {
  auto op = getOperation();
  scalelut_param_t   param = {0};
  param.shape_dim = 4;
  param.table_length = 256;
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_scalelut_global", &param, sizeof(param),
                           input_spec->data(), output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::ScaleLutOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice, group_type_t group_type) {
  return 0;
}

void tpu::ScaleLutOp::codegen_local_bm1684x(int64_t n_step, int64_t h_step,
                                            group_type_t group_type,
                                            local_sec_info_t &sec_info) {
  llvm_unreachable("Not Implemented");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================

int64_t tpu::ScaleLutOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(scalelut_param_t);
  scalelut_param_t   param = {0};
  param.shape_dim = 4;
  param.table_length = 256;
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::ScaleLutOp::get_layer_type() {
  return FW_BMNET_SCALE_LUT;
}
