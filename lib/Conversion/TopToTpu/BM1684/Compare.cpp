//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void CompareLowering::LoweringINT8(PatternRewriter &rewriter, top::CompareOp op,
                                   bool asymmetric) const {
  auto op_ = op.getOperation();
  double l_scale, r_scale;
  int64_t l_zp = 0, r_zp = 0;
  auto lhs = op.getLhs().getDefiningOp();
  auto rhs = op.getRhs().getDefiningOp();
  if (auto lhs_weight = dyn_cast<top::WeightOp>(lhs)) {
    if (lhs_weight.getScale().has_value()) {
      auto weight_scale_v = module::getF64Array(lhs_weight.getScale().value());
      l_scale = weight_scale_v->data()[0];
    } else {
      auto weight_f32 = lhs_weight.read<float>();
      double w_max = findMaxabs(weight_f32->data(), weight_f32->size());
      l_scale = w_max / 127.0;
    }
  } else {
    module::getScaleAndZeroPoint(op.getLhs(), l_scale, l_zp, asymmetric);
  }
  if (auto rhs_weight = dyn_cast<top::WeightOp>(rhs)) {
    if (rhs_weight.getScale().has_value()) {
      auto weight_scale_v = module::getF64Array(rhs_weight.getScale().value());
      r_scale = weight_scale_v->data()[0];
    } else {
      auto weight_f32 = rhs_weight.read<float>();
      double w_max = findMaxabs(weight_f32->data(), weight_f32->size());
      r_scale = w_max / 127.0;
    }
  } else {
    module::getScaleAndZeroPoint(op.getRhs(), r_scale, r_zp, asymmetric);
  }
  if (l_scale != r_scale || l_zp != r_zp) {
    lowering_common_f32<tpu::CompareOp>(rewriter, op_);
    return;
  }

  auto newType = getQuantBoolType(op.getOutput());
  lowering_common<tpu::CompareOp>(rewriter, op_, newType);
}

void CompareLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::CompareOp op) const {
  lowering_common_f32<tpu::CompareOp>(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
