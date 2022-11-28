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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::LutOp::init(InferenceParameter &p) { return success(); }
void tpu::LutOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LutOp::inference(InferenceParameter &p) {
  auto num_element = Module::getNumElements(input());
  auto chip = Module::getChip(getOperation());
  bool is_cv18xx = Module::isCV18xx(chip);
  if (Quant::isUniformQuantized(input())) {
    auto stype = Module::getStorageType(input());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
    for (int i = 0; i < num_element; ++i) {
      int offset = p.inputs[0][i];
      if (offset < 0) {
        offset += 256;
      }
      assert(offset >= 0 && offset <= 255);
      p.outputs[0][i] = p.inputs[1][offset];
    }
  } else {
    if (is_cv18xx) {
      auto _lut_mode = lut_mode();
      if (_lut_mode == LutMode::Slope) {
        bf16_lut_slope(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                      p.inputs[2], min_range().convertToDouble(),
                      max_range().convertToDouble());
      } else if (_lut_mode == LutMode::Mantissa) {
        bf16_lut_mantissa(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                      p.inputs[2], "mantissa");
      } else if (_lut_mode == LutMode::Log) {
        bf16_lut_mantissa(p.inputs[0], p.outputs[0], num_element, p.inputs[1],
                      p.inputs[2], "log");
      } else {
        llvm_unreachable("Not supported now!");
      }
    } else {
      llvm_unreachable("not support");
    }
  }
  return success();
}
