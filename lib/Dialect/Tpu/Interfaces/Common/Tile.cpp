//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::TileOp::init(InferenceParameter &p) { return success(); }
void tpu::TileOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::TileOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto out_shape = Module::getShape(output());
  auto in_shape = Module::getShape(input());
  auto axis_ = axis();
  int tile_ = tile();
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
  return success();
}
