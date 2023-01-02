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

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"



LogicalResult tpu::MulConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MulConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] * getConstVal().convertToDouble();
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum =
            applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(), getRshift());
        if (getDoRelu() && sum < 0)
          sum = 0;
        p.outputs[0][i] = out_type.isUnsignedInteger(8) ? to_uint8(sum)
                                                        : to_int8(sum);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.inputs[0][i];
        if (getDoRelu() && sum < 0)
          sum = 0;
        p.outputs[0][i] = out_type.isUnsignedInteger(8) ? to_uint8(sum)
                                                        : to_int8(sum);
      }
    }
  }
  return success();
}
