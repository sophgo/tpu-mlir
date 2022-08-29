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
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::MulOp::init(InferenceParameter &p) { return success(); }

void tpu::MulOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  int nInputs = inputs().size();
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  auto asym = Module::getAsymmetric(module);
  if (out_type.isa<FloatType>()) {
    std::fill(p.outputs[0], p.outputs[0] + num_elem, 1.0f);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t j = 0; j < num_elem; j++) {
      for (int i = 0; i < nInputs; i++) {
        p.outputs[0][j] *= p.inputs[i][j];
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
  } else if (asym == false) {
    memset(p.outputs[0], 0, num_elem * sizeof(float));
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int i = 0; i < num_elem; i++) {
      double sum = p.inputs[0][i] * p.inputs[1][i];
      sum = applyMultiplierAndRShift(sum, multiplier(), rshift());
      if (do_relu() && sum < 0) sum = 0;
      p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(sum)
                                                      : Quant::to_int8(sum);
    }
  } else {
    llvm_unreachable("MulOp asymmetric use FP32");
  }
  return success();
}
