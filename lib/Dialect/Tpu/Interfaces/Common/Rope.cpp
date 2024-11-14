//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <cstdint>
#include <limits>

template <typename T>
T saturate_num(int32_t value) {
  const T min_value = std::numeric_limits<T>::min();
  const T max_value = std::numeric_limits<T>::max();

  if (value < min_value) {
    return min_value;
  } else if (value > max_value) {
    return max_value;
  } else {
    return static_cast<T>(value);
  }
}

LogicalResult tpu::RopeOp::init(InferenceParameter &p) {

  auto binary = new Binary();
  p.handle = (void *)binary;
  return success();
}

void tpu::RopeOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto binary = (Binary *)p.handle;
    delete binary;
    p.handle = nullptr;
  }
}

LogicalResult tpu::RopeOp::inference(InferenceParameter &p) {

  auto num_element = module::getNumElements(getInput1());
  float *temp_input = new float[num_element];
  float *temp_result0 = new float[num_element];
  float *temp_result1 = new float[num_element];
  auto input_shape = module::getShape(getInput1());
  auto weight_shape = module::getShape(getInput2());
  auto binary = (Binary *)p.handle;
  auto out_type = module::getStorageType(getOutput());

#pragma omp parallel for schedule(static, omp_schedule(num_element))

  for (int i = 0; i < num_element; i++) {
    temp_input[i] = p.inputs[0][i];
  }

  for (int i = 0; i < num_element; i++) {
    if (i % 2 == 0 && i + 1 < num_element) {
      float temp = temp_input[i];
      temp_input[i] = temp_input[i + 1];
      temp_input[i + 1] = temp;
      temp_input[i] = -temp_input[i];
    }
  }

  float *weight0 = p.inputs[1];
  float *weight1 = p.inputs[2];
  float *input = p.inputs[0];
  int32_t mul1_shift = getMul1Shift();
  int32_t mul2_shift = getMul2Shift();
  int32_t add_shift = getAddShift();
  auto mul1_round_mode = round_mode_convert(getMul1RoundMode());
  auto mul2_round_mode = round_mode_convert(getMul2RoundMode());
  auto add_round_mode = round_mode_convert(getAddRoundMode());

  (*binary)
      .hs(temp_input, weight0, input_shape, weight_shape)
      .dst(temp_result0, module::getShape(getOutput()))
      .algorithem(algorithm::binary_mul)
      .setup();
  binary->run();

  if (out_type.isInteger(8) || out_type.isInteger(16) ||
      out_type.isInteger(32)) {
    for (int i = 0; i < num_element; i++) {
      int64_t sum_res0 = temp_result0[i];
      sum_res0 = RightShiftRound(sum_res0, -mul1_shift, mul1_round_mode);
      if (out_type.isInteger(8))
        temp_result0[i] = saturate_num<int16_t>(sum_res0);
      else if (out_type.isInteger(16))
        temp_result0[i] = saturate_num<int32_t>(sum_res0);
      else
        temp_result0[i] = saturate_num<int32_t>(sum_res0);
    }
  }

  (*binary)
      .hs(input, weight1, input_shape, weight_shape)
      .dst(temp_result1, module::getShape(getOutput()))
      .algorithem(algorithm::binary_mul)
      .setup();
  binary->run();
  if (out_type.isInteger(8) || out_type.isInteger(16) ||
      out_type.isInteger(32)) {
    for (int i = 0; i < num_element; i++) {
      int64_t sum_res1 = temp_result1[i];
      sum_res1 = RightShiftRound(sum_res1, -mul2_shift, mul2_round_mode);
      if (out_type.isInteger(8))
        temp_result1[i] = saturate_num<int16_t>(sum_res1);
      else if (out_type.isInteger(16))
        temp_result1[i] = saturate_num<int32_t>(sum_res1);
      else
        temp_result1[i] = saturate_num<int32_t>(sum_res1);
    }
  }

  for (int i = 0; i < num_element; i++) {
    p.outputs[0][i] = temp_result0[i] + temp_result1[i];
    if (out_type.isInteger(8) || out_type.isInteger(16) ||
        out_type.isInteger(32)) {
      int64_t sum = p.outputs[0][i];
      sum = RightShiftRound(sum, -add_shift, add_round_mode);
      p.outputs[0][i] = saturate(sum, out_type);
    }
  }

  delete[] temp_input;
  delete[] temp_result0;
  delete[] temp_result1;
  return success();
}
bool tpu::RopeOp::support_multi_core() { return false; }
