//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace mlir;

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) { return success(); }

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  auto axis_ = axis();
  auto input_shape = Module::getShape(input());
  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }

  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  int channel = input_shape[axis_];
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
  return success();
}
