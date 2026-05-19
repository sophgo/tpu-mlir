//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::Fp8MatMulOp::init(InferenceParameter &p) {
  return success();
}

void tpu::Fp8MatMulOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::Fp8MatMulOp::inference(InferenceParameter &p) {
  // dequant weight back to f16/ bf16
  auto weight_value = getWeight();
  auto weight_shape =
      weight_value.getType().cast<RankedTensorType>().getShape();
  int K = weight_shape[0];
  int N = weight_shape[1];
  auto in_shape = getInput().getType().cast<RankedTensorType>().getShape();
  int64_t M = 1;
  for (int i = 0; i < in_shape.size() - 1; i++) {
    M *= in_shape[i];
  }
  auto weight = p.inputs[1];
  auto scale = p.inputs[2];
  int block_size = getBlockSize();
  auto weight_transpose = getWeightTranspose();
  auto weight_len = K * N;
  auto new_weight = std::vector<float>(weight_len, 0);
  auto matmul = new MatMul();
  if (weight_transpose) {
    std::swap(K, N);
  }
  matmul->dequant_fp8_weight(new_weight.data(), weight, scale, weight_len,
                             block_size, N, K);
  matmul->setup(p.inputs[0], new_weight.data(), p.inputs[3], p.outputs[0], 1, 1,
                M, K, N, false, -1.0, 0, 0, weight_transpose, false, false,
                false);
  matmul->run();
  delete matmul;

  auto num_elem = module::getNumElements(getOutput());
  if (module::isF16Modes()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  } else {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

ArrayAttr tpu::Fp8MatMulOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  // TODO: Not support indexing maps for now
  return Builder(ctx).getAffineMapArrayAttr({});
}

bool tpu::Fp8MatMulOp::support_multi_core() {
  return (module::isSG2380() || module::isBM1690Family()) &&
         !module::isOpInGroupParallel(*this);
}
