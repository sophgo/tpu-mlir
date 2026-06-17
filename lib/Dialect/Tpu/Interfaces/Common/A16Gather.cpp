//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

LogicalResult tpu::A16GatherOp::init(InferenceParameter &p) {
  return success();
}

void tpu::A16GatherOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::A16GatherOp::inference(InferenceParameter &p) {
  auto ax = getAxis();
  assert(ax == 0 && "A16Gather only supports axis=0");

  auto weight_bits = getWeightBits();
  int q_group_size = getQGroupSize();

  // weight shape: [vocab_size, dim_packed]
  auto weight_shape = module::getShape(getWeight()).vec();
  int64_t vocab_size = weight_shape[0];
  int64_t dim = weight_shape[1] * 8 / weight_bits;

  // Chip-specific dim limitations
  if (module::isBM1688()) {
    assert(dim <= 1280 && "A16Gather: dim exceeds bm1688 limit (1280)");
  } else if (module::isBM1684XFamily()) {
    assert(dim <= 2560 && "A16Gather: dim exceeds bm1684x limit (2560)");
  }

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
        dequant_weight[idx] = ((int(weight[idx]) - zp_i) * scale_i);
      }
    }
  }

  // Gather: output[i, :] = dequant_weight[indices[i], :]
  float *inds = indices;
  float *dst = p.outputs[0];
  auto num_indices = module::getNumElements(getIndices());

#pragma omp parallel for schedule(static, omp_schedule(num_indices *dim))
  for (int64_t i = 0; i < num_indices; ++i) {
    int64_t idx = (int64_t)(inds[i] < 0 ? inds[i] + vocab_size : inds[i]);
    assert(idx >= 0 && idx < vocab_size && "A16Gather index out of range");
    memcpy(dst + i * dim, dequant_weight.data() + idx * dim,
           dim * sizeof(float));
  }

  // Convert output to target precision (BF16 or F16)
  auto num_elem = module::getNumElements(getOutput());
  if (module::isF16Modes()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  } else {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

mlir::Type tpu::A16GatherOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // indices must be integer type
    auto opd = op->getOperand(1);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwidth = stype.getIntOrFloatBitWidth();
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      bitwidth = 32;
    }
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

ArrayAttr tpu::A16GatherOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  int indices_dims = module::getShape(getIndices()).size();
  AffineMap indiceMap = AffineMap::getMultiDimIdentityMap(indices_dims, ctx);
  auto empty_map = AffineMap::get(2, 0, ctx);
  // keepdims=false: output has indices_dims + 1 dims (indices_shape + dim)
  // keepdims=true:  output has indices_dims + 2 dims (indices_shape + 1 + dim)
  int out_dims = indices_dims + (getKeepdims() ? 2 : 1);
  AffineMap outMap = AffineMap::getMultiDimIdentityMap(out_dims, ctx);
  SmallVector<AffineMap> indexingMaps{empty_map, indiceMap, empty_map,
                                      empty_map, outMap};
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::A16GatherOp::support_multi_core() { return false; }
