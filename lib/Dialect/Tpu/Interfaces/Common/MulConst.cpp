//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

LogicalResult tpu::MulConstOp::init(InferenceParameter &p) { return success(); }

void tpu::MulConstOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::MulConstOp::inference(InferenceParameter &p) {
  auto output_shape = computer_broadcast_shape(getOperation());
  module::setShape(getOutput(), output_shape);
  auto num_elem = module::getNumElements(getOutput());
  auto out_type = module::getStorageType(getOutput());
  auto asym = module::isAsymmetric();
  double const_v = getConstVal().convertToDouble();
  if (out_type.isFloat8E4M3FN()) {
    const_v = F16(const_v, true);
  }
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = p.inputs[0][i] * const_v;
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
    if (asym == false) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // coeff has been merge in multiplier&&rshift
        double sum = applyMultiplierAndRShift(p.inputs[0][i], getMultiplier(),
                                              getRshift());
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int i = 0; i < num_elem; i++) {
        // inputs has been requant
        double sum = p.inputs[0][i];
        if (getDoRelu() && sum < 0) {
          sum = 0;
        }
        p.outputs[0][i] = saturate(sum, out_type);
      }
    }
  }
  return success();
}

LogicalResult tpu::MulConstOp::LocalGenSupport() { return success(); }

void tpu::MulConstOp::assign_fw_param(void *param) {
  IR_PARAM_CONST_BINARY(BINARY_MUL);
}

mlir::Type tpu::MulConstOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  // int32-mul-float is not support. C is float point currently.
  // input/C/output must be all integer or all float point
  auto op = getOperation();
  // if (op && opd_idx == 0) {
  //   auto opd = op->getOperand(0);
  //   auto in_op = opd.getDefiningOp();
  //   if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
  //     return do_nothing(mode);
  //   }
  //   auto stype = module::getStorageType(opd);
  //   if (stype.isInteger(32)) {
  //     mode = TypeCastMode::DO_CAST;
  //     return Builder(op).getF32Type();
  //   }
  // }
  // cast input according to output type directly
  return type_verify_case_same(op, opd_idx, mode);
}

ArrayAttr tpu::MulConstOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::MulConstOp::support_multi_core() { return false; }
