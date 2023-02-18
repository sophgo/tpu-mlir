//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"



int64_t top::TileOp::getFLOPs() { return 0; }

LogicalResult top::TileOp::init(InferenceParameter &p) { return success(); }
void top::TileOp::deinit(InferenceParameter &p) {}

LogicalResult top::TileOp::inference(InferenceParameter &p) {
  auto out_shape = module::getShape(getOutput());
  auto in_shape = module::getShape(getInput());
  auto signed_axis = getAxisAttr().getValue().getSExtValue();
  auto axis_ = signed_axis > 0 ? signed_axis : 0;
  int tile_ = getTile();
  auto outer_count = std::accumulate(in_shape.begin(), in_shape.begin() + axis_,
                                     1, std::multiplies<int64_t>());
  auto inner_count = std::accumulate(in_shape.begin() + axis_, in_shape.end(),
                                     1, std::multiplies<int64_t>());
  auto input = p.inputs[0];
  auto output = p.outputs[0];
#pragma omp parallel for schedule(static, omp_schedule(outer_count))
  for (int out = 0; out < outer_count; ++out) {
    auto start = input + out * inner_count;
    auto end = start + inner_count;
    for (int t = 0; t < tile_; ++t) {
      std::copy(start, end,
                output + out * tile_ * inner_count + t * inner_count);
    }
  }
  return success();
}
