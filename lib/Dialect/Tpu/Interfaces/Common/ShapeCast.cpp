//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

using namespace tpu_mlir::backend;

LogicalResult tpu::ShapeCastOp::init(InferenceParameter &p) {
  return success();
}

void tpu::ShapeCastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeCastOp::inference(InferenceParameter &p) {
  const int num_elem = module::getNumElements(getInput());
  std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  return success();
}

bool tpu::ShapeCastOp::support_multi_core() { return false; }
