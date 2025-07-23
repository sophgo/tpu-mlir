//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

// ======================================
// GlobalGenInterface
// ======================================
void tpu::ShapeTileOp::codegen_global_bm1684x() {
  llvm_unreachable("Not supported now");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ShapeTileOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(shape_tile_param_t);
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  auto &in = input_spec->at(0);
  auto &out = output_spec->at(0);
  shape_tile_param_t spec = {0};
  for (int i = 0; i < in.dims; ++i) {
    spec.tile_coeff[i] = out.shape[i] / in.shape[i];
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ShapeTileOp::get_fw_type_bm1684x() { return FW_BMNET_SHAPE_TILE; }
