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

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::MulShiftOp::init(InferenceParameter &p) { return success(); }
void tpu::MulShiftOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulShiftOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  auto sType = Module::getStorageType(output());
  bool isUnsignInt = sType.isUnsignedInteger(8);
  int64_t in_zp = 0, out_zp = 0;
  if (Quant::isUniformQuantized(input())) {
    auto qtype = Quant::getUniformQuantizedType(input());
    in_zp = qtype.getZeroPoint();
  }
  if (Quant::isUniformQuantized(output())) {
    auto qtype = Quant::getUniformQuantizedType(output());
    out_zp = qtype.getZeroPoint();
  }

#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    auto v = applyMultiplierAndRShift(p.inputs[0][i] - in_zp,
                                      (int64_t)multiplier(), rshift());
    // should add zp to the outputs.
    v += out_zp;
    p.outputs[0][i] = isUnsignInt ? Quant::to_uint8(v) : Quant::to_int8(v);
  }
  return success();
}
