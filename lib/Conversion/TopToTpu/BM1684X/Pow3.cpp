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
// y = x1 ^ x2 = e ^ (x2 * log(x1))
// TODO: dangerous as need x1 > 0
void Pow3Lowering::LoweringF32(PatternRewriter &rewriter,
                               top::Pow3Op op) const {
  auto name = module::getName(op.getOutput());
  auto type = op.getOutput().getType();
  rewriter.setInsertionPointAfter(op);

  auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
  auto log_op = rewriter.create<tpu::ActiveOp>(
      log_loc, type, ValueRange{op->getOperand(0)}, attrs);

  auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
  attrs.clear();
  auto mul_op = rewriter.create<tpu::MulOp>(
      mul_loc, type, ValueRange{op->getOperand(1), log_op.getOutput()}, attrs);

  auto ex_loc = op.getLoc();
  attrs.clear();
  attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
  auto ex_op = rewriter.create<tpu::ActiveOp>(
      ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
  op.replaceAllUsesWith(ex_op.getOperation());
  rewriter.eraseOp(op);
}

void Pow3Lowering::LoweringINT8(PatternRewriter &rewriter, top::Pow3Op op,
                                bool asymmetric) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    LoweringBF16(rewriter, op);
  } else
    LoweringF32(rewriter, op);
}

void Pow3Lowering::LoweringINT4(PatternRewriter &rewriter, top::Pow3Op op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void Pow3Lowering::LoweringBF16(PatternRewriter &rewriter,
                                top::Pow3Op op) const {
  if (module::isMARS3() || module::isSGTPUV8()) {
    auto name = module::getName(op.getOutput());
    auto type = getQuantFloatType<BFloat16Type>(op.getOutput());
    rewriter.setInsertionPointAfter(op);

    auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
    auto log_op = rewriter.create<tpu::ActiveOp>(
        log_loc, type, ValueRange{op->getOperand(0)}, attrs);

    auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_mul"));
    attrs.clear();
    auto mul_op = rewriter.create<tpu::MulOp>(
        mul_loc, type, ValueRange{op->getOperand(1), log_op.getOutput()},
        attrs);

    auto ex_loc = op.getLoc();
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
    auto ex_op = rewriter.create<tpu::ActiveOp>(
        ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
    op.replaceAllUsesWith(ex_op.getOperation());
    rewriter.eraseOp(op);
  } else {
    LoweringF32(rewriter, op);
  }
}

void Pow3Lowering::LoweringF16(PatternRewriter &rewriter,
                               top::Pow3Op op) const {
  LoweringF32(rewriter, op);
}

void Pow3Lowering::LoweringF8(PatternRewriter &rewriter, top::Pow3Op op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void Pow3Lowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::Pow3Op op) const {
  llvm_unreachable("Not Implemented");
}
} // namespace bm1684x
} // namespace tpu_mlir
