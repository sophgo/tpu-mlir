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
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::PReluOp::init(InferenceParameter &p) {
  auto prelu = new PRelu();
  (*prelu)
      .src(p.inputs[0], Module::getShape(input()))
      .weights(p.inputs[1], Module::getShape(slope()))
      .dst(p.outputs[0], Module::getShape(output()))
      .setup();
  p.handle = (void *)prelu;
  return success();
}
void tpu::PReluOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto prelu = (PRelu *)p.handle;
    delete prelu;
    p.handle = nullptr;
  }
}

LogicalResult tpu::PReluOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }

  auto num_elem = Module::getNumElements(output());
  auto out_type = Module::getStorageType(output());
  bool asym = Module::isAsymmetric();
  bool is_cv18xx = Module::isCV18xx();
  if (out_type.isa<FloatType>()) {
    auto prelu = (PRelu *)p.handle;
    prelu->run();
    if (out_type.isBF16()) {
      f32_to_bf16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      f32_to_f16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (asym == false) {
    auto shift = rshift();
    int64_t shift_pos, multiplier_pos;
    if (is_cv18xx) {
      shift_pos = rshift_pos().value();
      multiplier_pos = muliplier_pos().value();
    }
    auto num_slope = Module::getNumElements(slope());
    auto in_shape = Module::getShape(input());
    int64_t num_inner = 1;
    int64_t num_outer = 1;
    if (in_shape.size() > 1) {
      num_outer = std::accumulate(in_shape.begin(), in_shape.begin() + 2, 1,
                                  std::multiplies<int64_t>());
      num_inner = std::accumulate(in_shape.begin() + 2, in_shape.end(), 1,
                                  std::multiplies<int64_t>());
    } else {
      num_outer = in_shape[0];
      num_inner = 1;
    }
#pragma omp parallel for schedule(static, omp_schedule(num_outer))
    for (int64_t i = 0; i < num_outer; i++) {
      int idx_slope = i % num_slope;
      int8_t slopei = p.inputs[1][idx_slope];
      for (int64_t j = 0; j < num_inner; j++) {
        int64_t idx = i * num_inner + j;
        if (is_cv18xx) {
          int64_t v;
          if (p.inputs[0][idx] < 0) {
            v = applyMultiplierAndRShift(p.inputs[0][idx], slopei, shift, CVI_QUANT);
          } else {
            v = applyMultiplierAndRShift(p.inputs[0][idx], multiplier_pos, shift_pos, CVI_QUANT);
          }
          p.outputs[0][idx] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                  : Quant::to_int8(v);
        } else {
          if (p.inputs[0][idx] < 0) {
            auto v = applyMultiplierAndRShift(p.inputs[0][idx], slopei, shift);
            p.outputs[0][idx] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(v)
                                                              : Quant::to_int8(v);
          } else {
            p.outputs[0][idx] = p.inputs[0][idx];
          }
        }
      }
    }
  } else {
    llvm_unreachable("Unknown type");
  }

  return success();
}
