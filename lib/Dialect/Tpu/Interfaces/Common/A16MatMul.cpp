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
#include "tpu_mlir/Support/Float4.h"
#include "tpu_mlir/Support/Float8.h"
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
  if (module::isDynamicQuantize()) {
    // dynamic group quantized matmul
    if (!q_group_size)
      return failure();
    if (getWeightBits() == 4) {
      N *= 2;
    }
    auto dynamic_quantize_type = getDqType();
    auto new_weight = std::vector<float>(K * N, 0);
    float imax;
    if (dynamic_quantize_type == "INT4" || dynamic_quantize_type == "INT8")
      imax = (1 << (getWeightBits() - 1)) - 1;
    else if (dynamic_quantize_type == "F8E4M3")
      imax = get_f8e4m3_max();
    // else if (dynamic_quantize_type == "F8E5M2")
    //   imax = get_f8e5m2_max();
    else if (dynamic_quantize_type == "F4")
      imax = get_f4e2m1_max();
    else
      return failure();
    if (getWeightBits() == 4) {
      for (int i = 0; i < K * N; i++) {
        int quant_idx = i / q_group_size;
        auto zp_i = zp[quant_idx];
        if (dynamic_quantize_type == "F4") {
          new_weight[i] = f4e2m1_to_f32(int(weight[i / 2]) & 0x0F) - zp_i;
          i++;
          new_weight[i] = f4e2m1_to_f32(int(weight[i / 2]) >> 4) - zp_i;
        } else {
          new_weight[i] = (int(weight[i / 2]) & 0x0F) - zp_i;
          i++;
          new_weight[i] = (int(weight[i / 2]) >> 4) - zp_i;
        }
      }
      if (w_transpose) {
        std::swap(K, N);
      }
    } else {
      std::swap(K, N);
      for (int i = 0; i < K * N; i++) {
        int quant_idx = i / q_group_size;
        auto zp_i = zp[quant_idx];
        if (dynamic_quantize_type == "F8E4M3") {
          new_weight[i] = f8e4m3_to_f32(weight[i]) - zp_i;
        // } else if (dynamic_quantize_type == "F8E5M2") {
        //   new_weight[i] = f8e5m2_to_f32(weight[i]) - zp_i;
        } else {
          new_weight[i] = int(weight[i]) - zp_i;
        }
      }
    }
    auto matmul = new MatMul();
    float *tmp_output = new float[M * N];
    float *cur_group_act = new float[M * q_group_size];
    float *cur_group_weight = new float[q_group_size * N];
    float *cur_group_act_max = new float[M];
    int group_num = (K + q_group_size - 1) / q_group_size;
    for (int gi = 0; gi < group_num; gi++) {
      for (int row = 0; row < M; row++) {
        cur_group_act_max[row] = 1e-8;
        for (int col = 0; col < q_group_size; col++) {
          if (gi * q_group_size + col < K) {
            cur_group_act_max[row] = std::max(
                cur_group_act_max[row],
                std::abs(p.inputs[0][row * K + gi * q_group_size + col]));
          }
        }
      }
      for (int row = 0; row < M; row++) {
        for (int col = 0; col < q_group_size; col++) {
          int act_idx = row * K + gi * q_group_size + col;
          if (gi * q_group_size + col < K) {
            if (dynamic_quantize_type == "INT4" ||
                dynamic_quantize_type == "INT8")
              cur_group_act[row * q_group_size + col] = std::round(
                  p.inputs[0][act_idx] / cur_group_act_max[row] * imax);
            else if (dynamic_quantize_type == "F8E4M3")
              cur_group_act[row * q_group_size + col] = F8E4M3(
                  p.inputs[0][act_idx], cur_group_act_max[row] / imax, true);
            // else if (dynamic_quantize_type == "F8E5M2")
            //   cur_group_act[row * q_group_size + col] = F8E5M2(
            //       p.inputs[0][act_idx], cur_group_act_max[row] / imax, true);
            else if (dynamic_quantize_type == "F4")
              cur_group_act[row * q_group_size + col] =
                  F4E2M1(p.inputs[0][act_idx], cur_group_act_max[row] / imax);
          } else {
            cur_group_act[row * q_group_size + col] = 0;
          }
        }
      }
      for (int row = 0; row < N; row++) {
        for (int col = 0; col < q_group_size; col++) {
          int weight_idx = row * K + gi * q_group_size + col;
          if (gi * q_group_size + col < K) {
            cur_group_weight[row * q_group_size + col] = new_weight[weight_idx];
          } else {
            cur_group_weight[row * q_group_size + col] = 0;
          }
        }
      }
      matmul->setup(cur_group_act, cur_group_weight, nullptr, tmp_output, 1, 1,
                    M, q_group_size, N, false, -1.0, 0, 0, w_transpose, false,
                    false, false);
      matmul->run();
      // accumulate result
      for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
          p.outputs[0][row * N + col] += tmp_output[row * N + col] *
                                         (cur_group_act_max[row] / imax) *
                                         scale[col * group_num + gi];
          if (gi == 0 && p.inputs[4] != nullptr) { // add bias
            p.outputs[0][row * N + col] += p.inputs[4][col];
          }
        }
      }
    }
    auto num_elem = module::getNumElements(getOutput());
    if (module::isF16Modes()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    }
    delete matmul;
    delete[] cur_group_act;
    delete[] cur_group_weight;
    delete[] cur_group_act_max;
    delete[] tmp_output;
    return success();
  } else {
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
      matmul->setup(p.inputs[0], new_weight.data(), p.inputs[4], p.outputs[0],
                    1, 1, M, K, N, false, -1.0, 0, 0, w_transpose, false, false,
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
      matmul->setup(p.inputs[0], new_weight.data(), p.inputs[4], p.outputs[0],
                    1, 1, M, K, N, false, -1.0, 0, 0, w_transpose, false, false,
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
  return failure();
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
