//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

LogicalResult tpu::SoftmaxBwdOp::init(InferenceParameter &p) {

  return success();
}

void tpu::SoftmaxBwdOp::deinit(InferenceParameter &p) {

}

LogicalResult tpu::SoftmaxBwdOp::inference(InferenceParameter &p) {

  return success();
}

uint32_t tpu::SoftmaxBwdOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}


int64_t tpu::SoftmaxBwdOp::get_fw_type_bm1684() {
  return -1;
}

