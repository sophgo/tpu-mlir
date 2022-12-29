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

#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"



LogicalResult tpu::LutBF16Op::init(InferenceParameter &p) { return success(); }
void tpu::LutBF16Op::deinit(InferenceParameter &p) {}

LogicalResult tpu::LutBF16Op::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(input());
  if (module::isCV18xx()) {
    auto _lut_mode = lut_mode();
    if (_lut_mode == LutBF16Mode::Slope) {
      bf16_lut_slope(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                     p.inputs[2], min_range().convertToDouble(),
                     max_range().convertToDouble());
    } else if (_lut_mode == LutBF16Mode::Mantissa) {
      bf16_lut_mantissa(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                        p.inputs[2], "mantissa");
    } else if (_lut_mode == LutBF16Mode::Log) {
      bf16_lut_mantissa(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                        p.inputs[2], "log");
    } else {
      llvm_unreachable("Not supported now!");
    }
  } else {
    llvm_unreachable("Only CV18XX support LutBF16Op");
  }
  return success();
}
