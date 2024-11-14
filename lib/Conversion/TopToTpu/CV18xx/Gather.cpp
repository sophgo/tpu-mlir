//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-gather"

namespace tpu_mlir {
namespace cv18xx {
void GatherLowering::LoweringINT8(PatternRewriter &rewriter, top::GatherOp op,
                                  bool asymmetric) const {
  LoweringBF16(rewriter, op);
}

void GatherLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::GatherOp op) const {
  std::vector<Value> operands;
  auto ishape = module::getShape(op.getInput());
  auto axis = op.getAxis();
  if (ishape.size() != 2 || axis != 0) {
    llvm_unreachable("Not support now.\n");
  }
  // auto inputOp = cast<top::WeightOp>(op.getInput().getDefiningOp());
  if (auto indiceOp =
          dyn_cast_or_null<top::WeightOp>(op.getIndices().getDefiningOp())) {
    operands.emplace_back(indiceOp.clone_int(op));
  } else {
    operands.emplace_back(op.getIndices());
  }
  // operands.emplace_back(op.getIndices());
  // operands.emplace_back(inputOp.clone_bf16(op));
  if (auto inputOp =
          dyn_cast_or_null<top::WeightOp>(op.getInput().getDefiningOp())) {
    operands.emplace_back(inputOp.clone_bf16(op));
  } else {
    operands.emplace_back(op.getInput());
  }
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getNamedAttr(
      "cpu_op_name", rewriter.getStringAttr("embedding")));
  attrs.emplace_back(
      rewriter.getNamedAttr("param", rewriter.getDictionaryAttr({})));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op.getOperation(), newType,
                                                 operands, attrs);
}
} // namespace cv18xx
} // namespace tpu_mlir
