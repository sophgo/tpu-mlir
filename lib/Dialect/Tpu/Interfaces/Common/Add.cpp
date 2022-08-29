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
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

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
  memset(p.outputs[0], 0, num_elem * sizeof(float));
  auto asym = Module::getAsymmetric(module);
  if (out_type.isa<FloatType>()) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t j = 0; j < num_elem; j++) {
      for (int i = 0; i < nInputs; i++) {
        p.outputs[0][j] += p.inputs[i][j];
      }
      if (do_relu()) {
        p.outputs[0][j] = std::max(0.0f, p.outputs[0][j]);
      }
    }
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (out_type.isInteger(32)) {
    // auto in0 = reinterpret_cast<int32_t*>(p.inputs[0]);
    // auto in1 = reinterpret_cast<int32_t*>(p.inputs[1]);
    // auto out = reinterpret_cast<int32_t*>(p.outputs[0]);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t j = 0; j < num_elem; j++) {
      for (int i = 0; i < nInputs; i++) {
        p.outputs[0][j] += p.inputs[i][j];
      }
      if (do_relu()) {
        p.outputs[0][j] = std::max(0.0f, p.outputs[0][j]);
      }
    }
  } else if (asym == false) {
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto zp = o_qtype.getZeroPoint();
    auto scale = o_qtype.getScale();
    auto chip = Module::getChip(module);
    auto op = getOperation();
    auto multiplier_v = Module::getI64Array(multipliers(), 2, 1);
    auto rshift_v = Module::getI64Array(rshifts(), 2, 0);
    memset(p.outputs[0], 0, num_elem * sizeof(float));
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      int64_t data0 = applyMultiplierAndRShift(
          p.inputs[0][i], multiplier_v->at(0), rshift_v->at(0));
      int64_t data1 = applyMultiplierAndRShift(
          p.inputs[1][i], multiplier_v->at(1), rshift_v->at(1));
      int64_t sum = data0 + data1;
      if (do_relu() && sum < 0) {
        sum = 0;
      }
      p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(sum)
                                                      : Quant::to_int8(sum);
    }
  } else {
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto zp = o_qtype.getZeroPoint();
    auto scale = o_qtype.getScale();
    auto op = getOperation();
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
      p.outputs[0][i] = p.outputs[0][i] / scale + zp;
      p.outputs[0][i] = out_type.isUnsignedInteger(8)
                            ? Quant::to_uint8(p.outputs[0][i])
                            : Quant::to_int8(p.outputs[0][i]);
    }
    return success();
  }

  return success();
}
