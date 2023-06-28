//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <valarray>


using namespace tpu_mlir::backend;

LogicalResult tpu::SqueezeOp::init(InferenceParameter &p) { return success(); }
void tpu::SqueezeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SqueezeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

LogicalResult tpu::SqueezeOp::LocalGenSupport() {
  if (module::isCV18xx() || module::isBM1684Family()) {
    return failure();
  }

  auto ishape = module::getShape(getInput());
  auto oshape = module::getShape(getOutput());
  if (ishape.size() < 2 || oshape.size() < 2 || ishape[0] != oshape[0] ||
      ishape[1] != oshape[1]) {
    return failure();
  }
  return success();
}
