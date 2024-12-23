//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/LutFunc.h"

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) { return success(); }

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  auto axis_ = getAxis();
  auto input_shape = module::getShape(getInput());
  module::setShape(getOutput(), input_shape);
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
  bool has_table = !module::isNone(getTable());
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
                       p.inputs[1], p.inputs[2], -EXP_BF16_LUT_RANGE,
                       EXP_BF16_LUT_RANGE);
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
    // auto round_mode = round_mode_convert(getRoundMode());
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
          auto offset =
              to_uint8(max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          sum += exp_table[offset];
        }
        for (int c = 0; c < channel; ++c) {
          auto offset =
              to_uint8(max_val - p.inputs[0][out_offset + c * inner_dim + j]);
          float prob_rescaled = exp_table[offset];
          prob_rescaled = prob_rescaled / (sum * scale);
          if (out_type.isSignedInteger(8)) {
            int prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
            p.outputs[0][out_offset + c * inner_dim + j] =
                to_int8(prob_rnd + zp);
          } else if (out_type.isUnsignedInteger(8)) {
            int prob_rnd = static_cast<int32_t>(prob_rescaled + 0.5);
            p.outputs[0][out_offset + c * inner_dim + j] =
                to_uint8(prob_rnd + zp);
          } else {
            llvm_unreachable("not support type");
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
    if (o_stype.isF32() && i_stype.isF32()) {
      if (module::isAsymmetric() == false) {
        return do_nothing(mode);
      }
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

LogicalResult tpu::SoftmaxOp::LocalGenSupport() {
  if (module::isCV18xx() || module::isBM1684Family()) {
    return failure();
  }
  int axis = getAxis();
  auto shape = module::getShape(getOutput());
  if (shape.size() == 4 && (axis == 2 || axis == 3)) {
    return success();
  }
  return failure();
}

LogicalResult tpu::SoftmaxOp::AllowDataSplit(int64_t axis,
                                             group_type_t group_type) {
  int64_t ax = getAxis();
  if (group_type == GROUP_SMALL_C) {
    ax = 2;
  }
  return axis < ax ? success() : failure();
}

ArrayAttr tpu::SoftmaxOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  int axis = getAxis();

  auto inputMap = AffineMap::getMultiDimIdentityMap(axis, ctx);
  auto empty = AffineMap::get(axis, 0, ctx);
  SmallVector<AffineMap> indexingMaps{inputMap};
  for (int i = 1, n = getNumOperands(); i < n; ++i) {
    indexingMaps.push_back(empty);
  }
  indexingMaps.push_back(inputMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::SoftmaxOp::support_multi_core() { return module::isSG2380(); }
