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

void CompareConstLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  lowering_common_f32<tpu::CompareConstOp>(rewriter, op);
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

} // namespace bm1684
} // namespace tpu_mlir
