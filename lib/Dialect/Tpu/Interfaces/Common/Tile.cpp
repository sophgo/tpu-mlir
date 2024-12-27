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
  auto tmp_size = in_shape.size();
  std::vector<int64_t> tile_vec;
  if (!getTileT()) {
    auto tile_v = module::getI64Array(getTile());
    tile_vec = *tile_v;
  } else {
    for (int i = 0; i < tmp_size; ++i) {
      tile_vec.emplace_back((int64_t)p.inputs[1][i]);
    }
  }

  int last_i = tile_vec.size() - 1;
  for (int i = 0; i < tile_vec.size(); ++i) {
    last_i = tile_vec.size() - i - 1;
    if (tile_vec[last_i] != 1)
      break;
  }

  auto last_op = p.inputs[0];
  std::vector<int64_t> tmp_shape(in_shape.begin(), in_shape.end());

  for (int i = 0; i < last_i + 1; ++i) {
    if (tile_vec[i] == 1)
      continue;
    int len = std::accumulate(tmp_shape.begin(), tmp_shape.end(), 1,
                              std::multiplies<int64_t>());
    float *cur_input = new float[len];
    std::copy(last_op, last_op + len, cur_input);

    function_tile(cur_input, p.outputs[0],
                  llvm::ArrayRef<int64_t>(tmp_shape.data(), tmp_size), i,
                  tile_vec[i]);
    last_op = p.outputs[0];
    tmp_shape[i] *= tile_vec[i];
    delete[] cur_input;
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

bool tpu::TileOp::support_multi_core() { return false; }
