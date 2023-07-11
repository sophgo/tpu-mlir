//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx_global_api.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::ArgOp::codegen_global_cv18xx(int64_t layer_id) {
  auto shape = module::getShape(getInput());
  assert(!module::isUniformQuantized(getValues()) && "Not support int8 Clip.");

  uint64_t axis = getAxis();
  int64_t outer = std::accumulate(shape.begin(), shape.begin() + axis, 1,
                                  std::multiplies<int64_t>());

  int64_t inner = shape[axis];
  gaddr_t ga_input = module::getAddress(getInput());
  gaddr_t ga_output = module::getAddress(getIndices());

  cvk_fmt_t fmt = CVK_FMT_I8;
  if (!module::isUniformQuantized(getIndices())) {
    fmt = CVK_FMT_BF16;
  }
  cvi_backend_tg_argmax_kernel(layer_id,  // layer_id
                               ga_input,  // gaddr_t ga_input[]
                               ga_output, // gaddr_t ga_output
                               outer, inner, 256, fmt);

  return;
}
