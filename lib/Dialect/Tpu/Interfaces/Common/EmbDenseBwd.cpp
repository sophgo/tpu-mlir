//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

LogicalResult tpu::EmbDenseBwdOp::init(InferenceParameter &p) {

  return success();
}

void tpu::EmbDenseBwdOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::EmbDenseBwdOp::inference(InferenceParameter &p) {

  return success();
}

uint32_t tpu::EmbDenseBwdOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::EmbDenseBwdOp::get_fw_type_bm1684() { return -1; }

bool tpu::EmbDenseBwdOp::support_multi_core() { return false; }
