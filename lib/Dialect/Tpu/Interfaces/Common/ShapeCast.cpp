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
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Dnnl/Concat.h"


using namespace tpu_mlir::backend;

LogicalResult tpu::ShapeCastOp::init(InferenceParameter &p) { return success(); }

void tpu::ShapeCastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ShapeCastOp::inference(InferenceParameter &p) {
  const int num_elem = module::getNumElements(getInput());
  std::copy(p.inputs[0], p.inputs[0] + num_elem, p.outputs[0]);
  return success();
}
