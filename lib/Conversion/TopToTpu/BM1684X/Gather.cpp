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

void GatherLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);
    if (auto wOp = in.getDefiningOp<top::WeightOp>()) {
      auto wtype = module::getStorageType(in);
      if (i == 1)
        operands.push_back(wOp.clone_int(op));
      else
        operands.push_back(in);
    } else {
      operands.push_back(in);
    }
  }
  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, op.getOutput().getType(), operands, op->getAttrs());
}

void GatherLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                                  bool asymmetric) const {
  lowering_common_f32<tpu::GatherOp>(rewriter, op);
}
void GatherLowering::LoweringINT4(PatternRewriter &rewriter, top::GatherOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void GatherLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::GatherOp op) const {
  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);
    if (auto wOp = in.getDefiningOp<top::WeightOp>()) {
      auto wtype = module::getStorageType(in);
      if (i == 1)
        operands.push_back(wOp.clone_int(op));
      else
        operands.push_back(wOp.clone_bf16(op));
    } else {
      operands.push_back(in);
    }
  }
  auto newType = getQuantFloatType<BFloat16Type>(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, newType, operands, op->getAttrs());
}

void GatherLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::GatherOp op) const {
  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);
    if (auto wOp = in.getDefiningOp<top::WeightOp>()) {
      auto wtype = module::getStorageType(in);
      if (i == 1)
        operands.push_back(wOp.clone_int(op));
      else
        operands.push_back(wOp.clone_f16(op));
    } else {
      operands.push_back(in);
    }
  }
  auto newType = getQuantFloatType<Float16Type>(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, newType, operands, op->getAttrs());
}

void GatherLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::GatherOp op) const {
  lowering_common<tpu::GatherOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
