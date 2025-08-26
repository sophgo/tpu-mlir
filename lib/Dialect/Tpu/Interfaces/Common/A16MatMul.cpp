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
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::A16MatMulOp::init(InferenceParameter &p) {
  return success();
}

void tpu::A16MatMulOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::A16MatMulOp::inference(InferenceParameter &p) {
  // dequant weight back to f16/ bf16
  auto scale = p.inputs[2];
  auto zp = p.inputs[3];
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
  int q_group_size =
      getQGroupSize() ? getQGroupSize() : module::getQuantGroupSize();
  auto w_transpose = getWTranspose();
  if (getWeightBits() == 4) {
    N *= 2;

    auto new_weight = std::vector<float>(K * N, 0);
    if (!q_group_size) {
      for (int i = 0; i < K; i++) {
        auto offset = i * N;
        auto zp_i = zp[i];
        auto scale_i = scale[i];
        for (int j = 0; j < N; j++) {
          new_weight[offset + j] =
              (((int(weight[(offset + j) / 2]) & 0x0F) - zp_i) * scale_i);
          j++;
          new_weight[offset + j] =
              (((int(weight[(offset + j) / 2]) >> 4) - zp_i) * scale_i);
        }
      }
    } else {
      for (int i = 0; i < K * N; i++) {
        int quant_idx = i / q_group_size;
        auto zp_i = zp[quant_idx];
        auto scale_i = scale[quant_idx];
        new_weight[i] = (((int(weight[i / 2]) & 0x0F) - zp_i) * scale_i);
        i++;
        new_weight[i] = (((int(weight[i / 2]) >> 4) - zp_i) * scale_i);
      }
    }

    auto matmul = new MatMul();
    if (w_transpose) {
      std::swap(K, N);
    }
    matmul->setup(p.inputs[0], new_weight.data(), p.inputs[4], p.outputs[0], 1,
                  1, M, K, N, false, -1.0, 0, 0, w_transpose, false, false,
                  false);
    matmul->run();
    delete matmul;
  } else {
    std::swap(K, N);
    auto new_weight = std::vector<float>(N * K, 0);
    if (!q_group_size) {
      for (int i = 0; i < N; i++) {
        auto offset = i * K;
        for (int j = 0; j < K; j++) {
          new_weight[offset + j] = (weight[offset + j] - zp[i]) * scale[i];
        }
      }
    } else {
      for (int i = 0; i < K * N; i++) {
        int quant_idx = i / q_group_size;
        auto zp_i = zp[quant_idx];
        auto scale_i = scale[quant_idx];
        new_weight[i] = (weight[i] - zp_i) * scale_i;
      }
    }
    if (module::isF16Modes()) {
      F16(new_weight.data(), new_weight.data(), N * K);
    } else {
      BF16(new_weight.data(), new_weight.data(), N * K);
    }
    auto matmul = new MatMul();
    matmul->setup(p.inputs[0], new_weight.data(), p.inputs[4], p.outputs[0], 1,
                  1, M, K, N, false, -1.0, 0, 0, w_transpose, false, false,
                  false);
    matmul->run();
    delete matmul;
  }

  auto num_elem = module::getNumElements(getOutput());
  if (module::isF16Modes()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  } else {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

ArrayAttr tpu::A16MatMulOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  if (getWeightBits() == 4 || getWTranspose() != true) {
    return Builder(ctx).getAffineMapArrayAttr({});
  }
  auto outShape = module::getShape(getOutput());
  auto num_dims = outShape.size();
  // TODO(pengchao.hu): Only Weight slice
  auto out_nums = module::getNumElements(getOutput());
  if (out_nums != outShape[num_dims - 1]) {
    return Builder(ctx).getAffineMapArrayAttr({});
  }
  AffineMap outMap = AffineMap::getMultiDimIdentityMap(num_dims, ctx);
  AffineMap inputMap = AffineMap::get(
      num_dims, 0, outMap.getResults().slice(0, num_dims - 1), ctx);
  AffineMap weightMap = AffineMap::get(
      num_dims, 0, outMap.getResults().slice(num_dims - 1, 1), ctx);
  AffineMap scaleMap = AffineMap::get(
      num_dims, 0, outMap.getResults().slice(num_dims - 1, 1), ctx);
  AffineMap emptyMap = AffineMap::get(num_dims, 0, ctx);
  SmallVector<AffineMap> indexingMaps{inputMap, weightMap, scaleMap, emptyMap};
  if (module::isNone(getBias())) {
    indexingMaps.push_back(emptyMap);
  } else {
    indexingMaps.push_back(scaleMap);
  }
  indexingMaps.push_back(outMap);
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::A16MatMulOp::support_multi_core() {
  return (module::isSG2380() || module::isBM1690Family()) &&
         !module::isOpInGroupParallel(*this);
}
