//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <set>
#include <sstream>

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace tpu {

struct ConvertReluLimitPattern : public RewritePattern {
  ConvertReluLimitPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    bool is_cv18xx = module::isCV18xx();
    if (isa<func::ReturnOp>(op)) {
      return failure();
    }
    if (is_cv18xx && op->hasTrait<trait::SupportFuseRelu>() &&
        module::getStorageType(op->getResult(0)).isBF16()) {
      auto max = op->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
      if (max == -1) {
        return failure();
      }
      auto op_name = module::getName(op).str();
      op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.));
      auto uses = op->getResult(0).getUses();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("min", rewriter.getF64FloatAttr(0.)));
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
            useOp->setOperand(i, newOp.output());
          }
        }
      }
      return success();
    } else {
      return failure();
    }
  }
};

class ReluLimitPass : public ConvertReluLimitBase<ReluLimitPass> {
public:
  ReluLimitPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertReluLimitPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createConvertReluLimit() {
  return std::make_unique<ReluLimitPass>();
}
} // namespace tpu
} // namespace tpu_mlir
