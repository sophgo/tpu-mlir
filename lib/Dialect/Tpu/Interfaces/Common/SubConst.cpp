//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::SubConstOp::init(InferenceParameter &p) { return success(); }

void tpu::SubConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SubConstOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();

  auto do_relu = getDoRelu();
  auto relu_limit = getReluLimitAttr().getValueAsDouble();

  if (getIsReverse()) {
    if (in_type.isFloat8E4M3FN()) {
      double scale = getF8Scale().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            getConstVal().convertToDouble() - p.inputs[0][i] * scale;
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        auto tmp_input =
            module::isUniformQuantized(getOutput())
                ? applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(), 0)
                : p.inputs[0][i];
        p.outputs[0][i] = getConstVal().convertToDouble() - tmp_input;
      }
    }
  } else {
    if (in_type.isFloat8E4M3FN()) {
      double scale = getF8Scale().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] =
            p.inputs[0][i] * scale - getConstVal().convertToDouble();
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        auto tmp_input =
            module::isUniformQuantized(getOutput())
                ? applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(), 0)
                : p.inputs[0][i];
        p.outputs[0][i] = tmp_input - getConstVal().convertToDouble();
      }
    }
  }
  if (do_relu) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] = std::max(p.outputs[0][i], 0.0f);
    }
    if (relu_limit > 0) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; i++) {
        p.outputs[0][i] = std::min(p.outputs[0][i], (float)relu_limit);
      }
    }
  }
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.outputs[0][i], 1, getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.outputs[0][i] + o_qtype.getZeroPoint();
        if (getDoRelu() && sum < o_qtype.getZeroPoint()) {
          sum = o_qtype.getZeroPoint();
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

ArrayAttr tpu::SubConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::SubConstOp::support_multi_core() { return false; }
