#include "Passes.h"

namespace mlir {
/// Check if a `t` is a `tensor` with zero extents.
static std::optional<RankedTensorType> isZeroExtent(Type t) {
  auto operandType = dyn_cast<RankedTensorType>(t);
  if (operandType &&
      llvm::any_of(operandType.getShape(), [](int64_t s) { return s == 0; })) {
    return operandType;
  }
  return std::nullopt;
}

/// Replace operands of the operation that have zero-extent tensors with
/// a `tensor.empty` op of the same type. This breaks dependencies between
/// different operations which can be handled subsequently.
struct ReplaceZeroExtentOperands : public RewritePattern {
  ReplaceZeroExtentOperands(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (isa<tensor::EmptyOp, tensor::DimOp>(op)) {
      return failure();
    }
    Location loc = op->getLoc();
    bool didUpdate = false;
    for (OpOperand &operand : op->getOpOperands()) {
      auto operandType = isZeroExtent(operand.get().getType());
      if (!operandType) {
        continue;
      }
      if (operand.get().getDefiningOp<tensor::EmptyOp>()) {
        continue;
      }
      Operation *owner = operand.getOwner();
      int operandNum = operand.getOperandNumber();
      auto shape = tensor::getMixedSizes(rewriter, loc, operand.get());
      auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, shape, operandType->getElementType());
      rewriter.updateRootInPlace(
          owner, [&]() { owner->setOperand(operandNum, emptyTensorOp); });
      didUpdate = true;
    }
    return success(didUpdate);
  }
};

/// Forward the destination of a `tensor.insert_slice` to its uses
/// if the source is zero-extent.
struct FoldZeroExtentInserts : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    if (!isZeroExtent(sliceOp.getSource().getType())) {
      return failure();
    }
    rewriter.replaceOp(sliceOp, sliceOp.getDest());
    return success();
  }
};

struct RemoveZeroExtentTensorsPass
    : RemoveZeroExtentTensorsBase<RemoveZeroExtentTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

void RemoveZeroExtentTensorsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  SmallVector<Operation *> opWithZeroExtentTensorOperands;
  SmallVector<tensor::InsertSliceOp> insertSliceOps;

  RewritePatternSet patterns(context);
  patterns.insert<FoldZeroExtentInserts, ReplaceZeroExtentOperands>(context);
  mlir::memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    funcOp->emitOpError("failed to run canonicalizations (proxy for DCE)");
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass() {
  return std::make_unique<RemoveZeroExtentTensorsPass>();
}
} // namespace mlir
