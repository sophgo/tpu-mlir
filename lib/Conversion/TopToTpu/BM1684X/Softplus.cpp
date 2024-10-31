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

void SoftplusLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::SoftplusOp op) const {
  auto op_ = op.getOperation();
  op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                tpu::ActiveMode::SOFT_PLUS));
  lowering_common_f32<tpu::ActiveOp>(rewriter, op_);
}

void SoftplusLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::SoftplusOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void SoftplusLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::SoftplusOp op, bool asymmetric) const {
  Value table = create_lookup_table(
      op.getInput(), op.getOutput(), asymmetric,
      [](double val) { return val >20 ? val : std::log(std::exp(val) + 1); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void SoftplusLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::SoftplusOp op) const {
    if(module::isMARS3()){
      auto op_ = op.getOperation();
      op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
                                                  tpu::ActiveMode::SOFT_PLUS));
      lowering_common_bf16<tpu::ActiveOp>(rewriter, op_);
    } else {
      LoweringF32(rewriter, op);
    }

}

void SoftplusLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::SoftplusOp op) const {
  LoweringF32(rewriter, op);
  // uncomment when needed
  // auto op_ = op.getOperation();
  //     op_->setAttr("mode", tpu::ActiveModeAttr::get(op.getContext(),
  //                                               tpu::ActiveMode::SOFT_PLUS));
  // lowering_common_f16<tpu::ActiveOp>(rewriter, op_);
}

void SoftplusLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::SoftplusOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void SoftplusLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::SoftplusOp op) const {
  Value table =
      create_lookup_table(op.getInput(), op.getOutput(), true, [](double val) {
        return std::log(std::exp(val) + 1);
      });
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput(), table});
}

} // namespace bm1684x
} // namespace tpu_mlir
