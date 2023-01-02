//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void CompareConstLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  lowering_common_f32<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringINT4(PatternRewriter &rewriter, top::CompareConstOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CompareConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::CompareConstOp op,
                                        bool asymmetric) const {
  auto op_ = op.getOperation();
  int64_t zp;
  double scale;
  bool sign;
  module::getScaleAndZeroPoint(op.getInput(), scale, zp, sign, asymmetric);
  auto val = op.getConstVal().convertToDouble();
  double new_val = std::round(val / scale + zp);
  new_val = sign ? to_int8(new_val) : to_uint8(new_val);
  op_->setAttr("const_val", rewriter.getF64FloatAttr(new_val));
  auto newType = getQuantBoolType(op.getOutput());
  lowering_common<tpu::CompareConstOp>(rewriter, op.getOperation(), newType);
}

void CompareConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::CompareConstOp op) const {
  lowering_common_bf16<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  lowering_common_f16<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::CompareConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
