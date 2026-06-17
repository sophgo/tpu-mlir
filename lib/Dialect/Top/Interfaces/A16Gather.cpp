//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "cnpy.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::A16GatherOp::getFLOPs() { return 0; }

LogicalResult top::A16GatherOp::init(InferenceParameter &p) {
  return success();
}

void top::A16GatherOp::deinit(InferenceParameter &p) {}

LogicalResult top::A16GatherOp::inference(InferenceParameter &p) {
  auto ax = getAxis();
  assert(ax == 0 && "A16Gather only supports axis=0");

  auto weight_bits = getWeightBits();
  int q_group_size = getQGroupSize();

  // weight shape: [vocab_size, dim_packed]
  // For weight_bits=4, dim_packed = dim / 2 (two 4-bit values per byte)
  // For weight_bits=8, dim_packed = dim
  auto weight_shape = module::getShape(getWeight()).vec();
  int64_t vocab_size = weight_shape[0];
  int64_t dim = weight_shape[1] * 8 / weight_bits;

  if (q_group_size < 1) {
    q_group_size = dim;
  }

  int weight_len = vocab_size * dim;
  int n_groups = dim / q_group_size;

  // Dequantize weight table to float
  auto dequant_weight = std::vector<float>(weight_len, 0);
  auto weight = p.inputs[0];  // weight is the first operand
  auto indices = p.inputs[1]; // indices is the second operand
  auto scale = p.inputs[2];
  auto zp = p.inputs[3];

  if (weight_bits == 4) {
    // 4-bit packed: two values per byte
    for (int row = 0; row < vocab_size; row++) {
      for (int col = 0; col < dim; col += 2) {
        int packed_idx = row * (dim / 2) + col / 2;
        int unpacked_idx = row * dim + col;
        int group_idx = col / q_group_size;
        int quant_idx = row * n_groups + group_idx;

        auto zp_i = zp[quant_idx];
        auto scale_i = scale[quant_idx];

        // Unpack low nibble
        dequant_weight[unpacked_idx] =
            (((int(weight[packed_idx]) & 0x0F) - zp_i) * scale_i);
        // Unpack high nibble
        dequant_weight[unpacked_idx + 1] =
            (((int(weight[packed_idx]) >> 4) - zp_i) * scale_i);
      }
    }
  } else {
    // 8-bit: one value per byte
    for (int row = 0; row < vocab_size; row++) {
      for (int col = 0; col < dim; col++) {
        int idx = row * dim + col;
        int group_idx = col / q_group_size;
        int quant_idx = row * n_groups + group_idx;

        auto zp_i = zp[quant_idx];
        auto scale_i = scale[quant_idx];
        auto weight_i = weight[idx];

        dequant_weight[idx] = ((int(weight_i) - zp_i) * scale_i);
      }
    }
  }

  // Gather: output[i, j, :] = dequant_weight[indices[i, j], :]
  float *inds = indices;
  float *dst = p.outputs[0];
  auto num_indices = module::getNumElements(getIndices());
  int64_t total = num_indices * dim;

#pragma omp parallel for schedule(static, omp_schedule(total))
  for (int64_t i = 0; i < num_indices; ++i) {
    int64_t idx = (int64_t)(inds[i] < 0 ? inds[i] + vocab_size : inds[i]);
    assert(idx >= 0 && idx < vocab_size && "A16Gather index out of range");
    memcpy(dst + i * dim, dequant_weight.data() + idx * dim,
           dim * sizeof(float));
  }
  return success();
}

void top::A16GatherOp::shape_inference() {
  auto ax = getAxis();
  assert(ax == 0 && "A16Gather only supports axis=0");

  auto weight_shape = module::getShape(getWeight()).vec();
  auto indices_shape = module::getShape(getIndices()).vec();
  auto weight_bits = getWeightBits();
  int64_t dim = weight_shape[1] * 8 / weight_bits;

  // Output shape:
  //   keepdims=false: [*indices_shape, dim]
  //   keepdims=true:  [*indices_shape, 1, dim]
  std::vector<int64_t> out_shape = indices_shape;
  if (getKeepdims()) {
    out_shape.push_back(1);
  }
  out_shape.push_back(dim);
  module::setShapeOrVerify(getOutput(), out_shape);
}
