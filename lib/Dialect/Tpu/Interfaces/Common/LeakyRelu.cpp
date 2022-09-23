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

LogicalResult tpu::LeakyReluOp::init(InferenceParameter &p) {
  return success();
}
void tpu::LeakyReluOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::LeakyReluOp::inference(InferenceParameter &p) {
  int64_t num_elements = Module::getNumElements(input());
  memset(p.outputs[0], 0, sizeof(float) * num_elements);
  auto out_type = Module::getStorageType(output());
  auto module = Module::getModuleOp(getOperation());
  auto asym = Module::getAsymmetric(module);

  if (out_type.isa<FloatType>()) {
    float *src = p.inputs[0];
    float *dst = p.outputs[0];
    float alpha = static_cast<float>(alphaAttr().getValueAsDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      dst[i] =
          src[i] > 0
              ? src[i]
              : (alpha * src[i]);
    }
    if (out_type.isF16()) {
      f32_to_f16(dst, dst, num_elements);
    } else if (out_type.isBF16()) {
      f32_to_bf16(dst, dst, num_elements);
    }
  } else if (asym == false) {
    int64_t scalei = multiplier().value();
    int64_t shifti = rshift().value();

#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      int64_t dst = 0;
      int64_t src = static_cast<int64_t>(p.inputs[0][i]);
      dst = src >= 0 ? src : applyMultiplierAndRShift(src, scalei, shifti);

      p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(dst)
                                                      : Quant::to_int8(dst);
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
    for (int64_t i = 0; i < num_elements; ++i) {
      int64_t src = static_cast<int64_t>(p.inputs[0][i]);
      int64_t dst = 0;

      auto i_qtype = Quant::getUniformQuantizedType(input());
      auto o_qtype = Quant::getUniformQuantizedType(output());
      double scale = i_qtype.getScale() / o_qtype.getScale();
      dst = src >= i_qtype.getZeroPoint()
                ? src
                : ((src - i_qtype.getZeroPoint()) * scale +
                   o_qtype.getZeroPoint());

      p.outputs[0][i] = out_type.isUnsignedInteger(8) ? Quant::to_uint8(dst)
                                                      : Quant::to_int8(dst);
    }
  }
  return success();
}
