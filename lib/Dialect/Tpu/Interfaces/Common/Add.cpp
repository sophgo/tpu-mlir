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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::AddOp::init(InferenceParameter &p) { return success(); }
void tpu::AddOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AddOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  int nInputs = inputs().size();
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  if (out_type.isF32() || out_type.isBF16() || out_type.isF16()) {
    memset(p.outputs[0], 0, num_elem * sizeof(float));

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t j = 0; j < num_elem; j++) {
      for (int i = 0; i < nInputs; i++) {
        p.outputs[0][j] += p.inputs[i][j];
      }
      if (do_relu()) {
        p.outputs[0][j] = std::max(0.0f, p.outputs[0][j]);
      }
    }
  } else {
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto zp = o_qtype.getZeroPoint();
    auto scale = o_qtype.getScale();
    auto b = rectified_bias().convertToDouble();
    auto chip = Module::getChip(module);
    bool debug_mode = false; // 1686 use debug mode: fp32 calc

    if (chip == Module::Chip::BM1686) {
      debug_mode = true;
    }

    if (debug_mode) {
      auto op = getOperation();
      for (int i = 0; i < num_elem; i++) {
        p.outputs[0][i] = 0;
      }
      for (int i = 0; i < nInputs; i++) {
        auto input = inputs()[i];
        auto qtype = Quant::getUniformQuantizedType(input);
        for (int64_t j = 0; j < num_elem; j++) {
          p.outputs[0][j] +=
              (p.inputs[i][j] - qtype.getZeroPoint()) * qtype.getScale();
        }
      }
      for (int i = 0; i < num_elem; i++) {
        if (do_relu()) {
          p.outputs[0][i] = std::max(0.0f, p.outputs[0][i]);
        }
        p.outputs[0][i] = Quant::to_int8(p.outputs[0][i] / scale + zp);
      }
      return success();
    }
    std::vector<float *> input_data = p.inputs;
    auto multiplier_v = Module::getI64Array(multipliers(), nInputs, 1);
    auto rshift_v = Module::getI64Array(rshifts(), nInputs, 0);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = 0;
      for (int64_t j = 0; j < nInputs; j++) {
        int half_data = 1 << (rshift_v->at(j) - 1);
        p.outputs[0][i] += ((int64_t)(p.inputs[j][i] * multiplier_v->at(j)) + half_data) >>
                           rshift_v->at(j);
      }

      if (chip == Module::Chip::BM1686) {
        p.outputs[0][i] -= b;
      }

      if (do_relu()) {
        p.outputs[0][i] = p.outputs[0][i] > 0 ? p.outputs[0][i] : 0;
      }
      p.outputs[0][i] = Quant::to_int8(p.outputs[0][i] + zp);
    }
  }

  return success();
}
