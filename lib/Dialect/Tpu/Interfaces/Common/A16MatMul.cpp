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
  // (MxK), (KxN) matmul
  if (getWeightBits() == 8) {
    auto matmul = new MatMul();
    auto in_shape = getInput().getType().cast<RankedTensorType>().getShape();
    int64_t M = 1;
    for (int i = 0; i < in_shape.size() - 1; i++) {
      M *= in_shape[i];
    }
    auto weight_value = getWeight();
    auto weight_shape =
        weight_value.getType().cast<RankedTensorType>().getShape();

    auto w_transpose = getWTranspose();
    int K = w_transpose ? weight_shape[1] : weight_shape[0];
    int N = w_transpose ? weight_shape[0] : weight_shape[1];

    matmul->setup(p.inputs[0], p.inputs[1], p.inputs[4], p.outputs[0], 1, 1, M,
                  K, N, false, -1.0, 0, 0, w_transpose, false, false, false);
    p.handle = (void *)matmul;
  }
  return success();
}

void tpu::A16MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::A16MatMulOp::inference(InferenceParameter &p) {
  // dequant weight back to f16/ bf16
  auto scale = p.inputs[2];
  auto zp = p.inputs[3];
  auto weight_value = getWeight();
  auto weight_shape =
      weight_value.getType().cast<RankedTensorType>().getShape();
  int K = weight_shape[0];
  int N = weight_shape[1];
  auto weight = p.inputs[1];
  int q_group_size =
      getQGroupSize() ? getQGroupSize() : module::getQuantGroupSize();
  if (getWeightBits() == 4) {
    auto w_transpose = getWTranspose();
    N *= 2;
    auto in_shape = getInput().getType().cast<RankedTensorType>().getShape();
    int64_t M = 1;
    for (int i = 0; i < in_shape.size() - 1; i++) {
      M *= in_shape[i];
    }
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
    if (!q_group_size) {
      for (int i = 0; i < K; i++) {
        auto offset = i * N;
        for (int j = 0; j < N; j++) {
          weight[offset + j] = module::isSG2380()
                                   ? ((weight[offset + j]) * scale[i] - zp[i])
                                   : ((weight[offset + j]) * scale[i]);
        }
      }
    } else {
      for (int i = 0; i < K * N; i++) {
        int quant_idx = i / q_group_size;
        auto zp_i = zp[quant_idx];
        auto scale_i = scale[quant_idx];
        weight[i] = (weight[i] - zp_i) * scale_i;
      }
    }
    // hand over the rest work to onednn matmul
    if (p.handle == nullptr) {
      return failure();
    }
    auto matmul = (MatMul *)p.handle;
    matmul->run();
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
