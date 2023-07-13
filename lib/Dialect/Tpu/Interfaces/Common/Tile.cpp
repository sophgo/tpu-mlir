//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::TileOp::init(InferenceParameter &p) { return success(); }
void tpu::TileOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::TileOp::LocalGenSupport() { return failure(); }

LogicalResult tpu::TileOp::inference(InferenceParameter &p) {
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

void tpu::TileOp::assign_fw_param(void *param) {
  fw_tile_layer_param_t tile_param = {0};
  tile_param.input_is_coeff = module::isWeight(getInput());
  tile_param.type = 0;
  tile_param.coeff_is_fixed = !module::isWeight(getInput());
  tile_param.input_dims = module::getShape(getInput()).size();
  int tile_count = 0;
  for (int i = 0; i < tile_param.input_dims; ++i) {
    tile_param.input_shape[i] = module::getShape(getInput())[i];
    tile_param.tile_coeff[i] =
        module::getShape(getOutput())[i] / tile_param.input_shape[i];
    if (tile_param.tile_coeff[i] > 1)
      tile_count += 1;
  }
  tile_param.buffer_addr = 0;
  if (module::isUniformQuantized(getInput())) {
    uint64_t global_buffer_size = 0;
    BM1684::instance().dl_nodechip_tile_full_fix8b(
        0, 0, 0, &global_buffer_size, (uint32_t *)tile_param.input_shape,
        tile_param.tile_coeff, tile_param.input_dims,
        BM1684::getStoreMode(getInput()), BM1684::getStoreMode(getOutput()), 0,
        NULL);
    if (global_buffer_size)
      tile_param.buffer_addr = module::getAddress(getBuffer());
  } else if (tile_count > 1) {
    tile_param.buffer_addr = module::getAddress(getBuffer());
  }
  memcpy(param, &tile_param, sizeof(fw_tile_layer_param_t));
}
