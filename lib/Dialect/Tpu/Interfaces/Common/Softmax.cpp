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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace mlir;

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) { return success(); }

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  auto axis_ = axis();
  auto input_shape = Module::getShape(input());
  auto out_type = Module::getStorageType(output());
  auto num_elem = Module::getNumElements(output());

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  int channel = input_shape[axis_];
  if (out_type.isa<FloatType>()) {
    float max_arr[inner_dim];
    float sum_arr[inner_dim];
    float ex_arr[channel * inner_dim];
    float sub_arr[channel * inner_dim];

    auto bottom_data = p.inputs[0];
    auto top_data = p.outputs[0];

    for (int i = 0; i < outer_dim; ++i) {
      // find max value accross channel
      int c_offset = i * channel * inner_dim;
      memcpy(max_arr, bottom_data + c_offset, inner_dim * sizeof(float));
      for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
        for (int k = 0; k < inner_dim; k++) {
          if (max_arr[k] < bottom_data[c_offset + k])
            max_arr[k] = bottom_data[c_offset + k];
        }
      }

      // calculate exp(x)
      c_offset = i * channel * inner_dim;
      memset(sum_arr, 0, inner_dim * sizeof(float));
      for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
        for (int k = 0; k < inner_dim; k++) {
          sub_arr[j * inner_dim + k] = bottom_data[c_offset + k] - max_arr[k];
          top_data[c_offset + k] = std::exp(sub_arr[j * inner_dim + k]);
          sum_arr[k] += top_data[c_offset + k];
        }
      }

      c_offset = i * channel * inner_dim;
      for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
        for (int k = 0; k < inner_dim; k++) {
          top_data[c_offset + k] /= sum_arr[k];
        }
      }
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (Quant::isUniformQuantized(input(),
                                       output())) { // for quant softmax
    auto exp_table = p.inputs[1];
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto zp = o_qtype.getZeroPoint();
    auto scale = o_qtype.getScale();
    for (int i = 0; i < outer_dim; ++i) {
      int64_t out_offset = i * inner_dim * channel;
#pragma omp parallel for schedule(static, omp_schedule(inner_dim))
      for (int j = 0; j < inner_dim; ++j) {
        int max_val = p.inputs[0][out_offset + j];
        for (int c = 1; c < channel; ++c) {
          max_val = max_val > p.inputs[0][out_offset + c * inner_dim + j]
                        ? max_val
                        : p.inputs[0][out_offset + c * inner_dim + j];
        }
        float sum = 0.f;
        for (int c = 0; c < channel; ++c) {
          auto offset = Quant::to_uint8(
              max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          sum += exp_table[offset];
        }
        for (int c = 0; c < channel; ++c) {
          auto offset = Quant::to_uint8(
              max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          float prob_rescaled = exp_table[offset];
          prob_rescaled = prob_rescaled / (sum * scale);
          if (out_type.isInteger(8)) {
            int prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
            p.outputs[0][out_offset + c * inner_dim + j] =
                Quant::to_int8(prob_rnd + zp);
          } else {
            int prob_rnd = static_cast<int32_t>(prob_rescaled + 0.5);
            p.outputs[0][out_offset + c * inner_dim + j] =
                Quant::to_uint8(prob_rnd + zp);
          }
        }
      }
    }
  } else {
    dump();
    llvm_unreachable("not support type");
  }
  return success();
}
