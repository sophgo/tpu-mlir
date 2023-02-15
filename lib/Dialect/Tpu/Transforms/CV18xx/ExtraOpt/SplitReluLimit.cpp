//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/CV18xx/DoExtraOpt.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "cv18xx_patterns"
namespace tpu_mlir {
namespace cv18xx {

LogicalResult
SplitReluLimitPattern::matchAndRewrite(Operation *op,
                                       PatternRewriter &rewriter) const {
  rewriter.setInsertionPointAfter(op);
  if (isa<ReturnOp>(op)) {
    return failure();
  }
  if (op->hasTrait<trait::SupportFuseRelu>() &&
      module::getStorageType(op->getResult(0)).isBF16()) {
    auto max = op->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
    if (max == -1) {
      return failure();
    }
    auto op_name = module::getName(op).str();
    op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.));
    auto uses = op->getResult(0).getUses();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("min", rewriter.getF64FloatAttr(0.)));
    attrs.push_back(
        rewriter.getNamedAttr("max", rewriter.getF64FloatAttr(max)));
    auto tensor_type = op->getResult(0).getType().cast<RankedTensorType>();
    auto newType =
        RankedTensorType::get(tensor_type.getShape(), rewriter.getBF16Type());
    auto newOp = rewriter.create<tpu::ClipOp>(op->getLoc(), newType,
                                              op->getResults(), attrs);
    op->setLoc(NameLoc::get(rewriter.getStringAttr(op_name + "_0")));
    for (auto &use : uses) {
      auto useOp = use.getOwner();
      int32_t num = useOp->getNumOperands();
      for (int32_t i = 0; i < num; i++) {
        if (useOp->getOperand(i) == op->getResult(0)) {
          useOp->setOperand(i, newOp.getOutput());
        }
      }
    }
    return success();
  } else {
    return failure();
  }
}

} // namespace cv18xx
} // namespace tpu_mlir
