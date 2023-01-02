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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) { return success(); }

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  auto axis_ = getAxis();
  auto input_shape = module::getShape(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  bool is_cv18xx = module::isCV18xx();

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  int channel = input_shape[axis_];
  bool has_table = !getTable().getType().isa<mlir::NoneType>();
  if (out_type.isa<FloatType>()) {
    float scale = 1.0f;
    if (module::isUniformQuantized(getInput())) {
      auto qtype = module::getUniformQuantizedType(getInput());
      scale = qtype.getScale();
    }
    std::vector<float> max_arr(inner_dim);
    std::vector<float> sum_arr(inner_dim);
    std::vector<float> sub_arr(channel * inner_dim);

    const auto bottom_data = p.inputs[0];
    auto top_data = p.outputs[0];

    for (int i = 0; i < outer_dim; ++i) {
      // find max value accross channel
      int c_offset = i * channel * inner_dim;
      memcpy(max_arr.data(), bottom_data + c_offset, inner_dim * sizeof(float));
      for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
        for (int k = 0; k < inner_dim; k++) {
          if (max_arr[k] < bottom_data[c_offset + k])
            max_arr[k] = bottom_data[c_offset + k];
        }
      }
      c_offset = i * channel * inner_dim;
      if (is_cv18xx) {
        // calculate x - max
        for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
          for (int k = 0; k < inner_dim; k++) {
            auto idx = j * inner_dim + k;
            sub_arr[idx] = BF16(bottom_data[c_offset + k] - max_arr[k]);
          }
        }
        // e^x
        std::vector<float> ex_arr(channel * inner_dim);
        bf16_lut_slope(sub_arr.data(), ex_arr.data(), sub_arr.size(),
                       p.inputs[1], p.inputs[2], -15, 15);
        // sum of (e^x)
        float const_val = BF16(BF16(1.0 * channel) / channel);
        memset(sum_arr.data(), 0, inner_dim * sizeof(float));
        for (int j = 0; j < channel; ++j) {
          for (int k = 0; k < inner_dim; k++) {
            auto idx = j * inner_dim + k;
            sum_arr[k] += ex_arr[idx] * const_val;
          }
        }
        // convert to bf16
        BF16(sum_arr.data(), sum_arr.data(), sum_arr.size());

        std::string mehod = getLog() ? "log" : "mantissa";
        bf16_lut_mantissa(sum_arr.data(), sum_arr.data(), sum_arr.size(),
                          p.inputs[3], p.inputs[4], mehod);

        c_offset = i * channel * inner_dim;
        for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
          for (int k = 0; k < inner_dim; k++) {
            auto idx = j * inner_dim + k;
            if (getLog()) {
              top_data[c_offset + k] = sub_arr[idx] - sum_arr[k];
            } else {
              top_data[c_offset + k] = ex_arr[idx] * sum_arr[k];
            }
          }
        }
      } else {
        // calculate exp(x)
        memset(sum_arr.data(), 0, inner_dim * sizeof(float));
        for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
          for (int k = 0; k < inner_dim; k++) {
            sub_arr[j * inner_dim + k] =
                (bottom_data[c_offset + k] - max_arr[k]) * scale;
            top_data[c_offset + k] = std::exp(sub_arr[j * inner_dim + k]);
            sum_arr[k] += top_data[c_offset + k];
          }
        }

        c_offset = i * channel * inner_dim;
        for (int j = 0; j < channel; ++j, c_offset += inner_dim) {
          for (int k = 0; k < inner_dim; k++) {
            top_data[c_offset + k] /= sum_arr[k];
            if (getLog()) {
              top_data[c_offset + k] = std::log(top_data[c_offset + k]);
            }
          }
        }
      }
    }
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getInput(),
                                       getOutput())) { // for quant softmax
    assert(has_table == true);
    auto exp_table = p.inputs[1];
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto zp = o_qtype.getZeroPoint();
    float scale = o_qtype.getScale();
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
          auto offset = to_uint8(
              max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          sum += exp_table[offset];
        }
        for (int c = 0; c < channel; ++c) {
          auto offset = to_uint8(
              max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          float prob_rescaled = exp_table[offset];
          prob_rescaled = prob_rescaled / (sum * scale);
          if (out_type.isInteger(8)) {
            int prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
            p.outputs[0][out_offset + c * inner_dim + j] =
                to_int8(prob_rnd + zp);
          } else {
            int prob_rnd = static_cast<int32_t>(prob_rescaled + 0.5);
            p.outputs[0][out_offset + c * inner_dim + j] =
                to_uint8(prob_rnd + zp);
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

mlir::Type tpu::SoftmaxOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  auto i_stype = module::getStorageType(getInput());
  auto o_stype = module::getStorageType(getOutput());
  if (opd_idx == 0) {
    if (o_stype.isF32() && (i_stype.isInteger(8) || i_stype.isF32())) {
      if (module::isAsymmetric() == false) {
        return do_nothing(mode);
      }
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}
