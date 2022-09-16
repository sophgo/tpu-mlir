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

LogicalResult tpu::AddConstOp::init(InferenceParameter &p) { return success(); }

void tpu::AddConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::AddConstOp::inference(InferenceParameter &p) {
  auto module = Module::getModuleOp(getOperation());
  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  auto asym = Module::getAsymmetric(module);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] + const_val().convertToDouble();
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (Quant::isUniformQuantized(output())) {
    if (asym == false) {
  #pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.outputs[0][i], multiplier(), rshift());
        if (do_relu() && sum < 0) sum = 0;
        p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(sum)
                                                        : Quant::to_int8(sum);
      }
    } else {
  #pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.outputs[0][i];
        if (do_relu() && sum < 0) sum = 0;
        p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(sum)
                                                        : Quant::to_int8(sum);
      }
    }
  }
  return success();
}
