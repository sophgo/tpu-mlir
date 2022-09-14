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
  auto in_dims = in_shape.size();
  assert(in_dims == out_shape.size());

  int tile_count = 0;
  SmallVector<int64_t> tile_coeff(in_dims);
  for (size_t i = 0; i < in_dims; ++i) {
    tile_coeff[i] = out_shape[i] / in_shape[i];
    if (tile_coeff[i] != 1) {
      tile_count++;
    }
  }

  float *buf = new float[num_elem];
  float *real_in = p.inputs[0];
  float *real_out_arr[2] = {buf, p.outputs[0]};
  int index = tile_count & 0x1;
  float *real_out = real_out_arr[index];
  int64_t high = Module::getNumElements(input());
  int64_t low = 1;
  for (int i = in_dims - 1; i >=0; --i) {
    if (tile_coeff[i] == 1) {
      continue;
    }
    high /= in_shape[i];
    low  *= in_shape[i];

    for (int64_t j = 0; j < high; ++j) {
      const float *in_j = real_in + j * low;
      float *out_j = real_out + j * tile_coeff[i] * low;
      for (int64_t k = 0; k < tile_coeff[i]; ++k) {
        memcpy(out_j + k * low, in_j, low * sizeof(float));
      }
    }

    low *= tile_coeff[i];
    real_in = real_out;
    index = !index;
    real_out = real_out_arr[index];
  }

  delete[] buf;
  return success();
}
