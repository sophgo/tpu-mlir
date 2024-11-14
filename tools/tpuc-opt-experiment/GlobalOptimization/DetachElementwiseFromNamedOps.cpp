#include "Passes.h"

namespace mlir {
struct DetachElementwisePattern
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isaContractionOpInterface(linalgOp) &&
        !isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
      return failure();
    }
    if (!linalgOp.hasTensorSemantics())
      return failure();

    // Nothing to do if the output tensor operand is already a fill op.
    OpOperandVector outputOperands;
    if (!linalgOp.hasBufferSemantics()) {
      outputOperands = linalgOp.getDpsInitOperands();
    }
    // Right now all the cases we see have one output. This can be relaxed once
    // we see multiple output ops.
    if (outputOperands.size() != 1)
      return failure();
    Value outputOperand = outputOperands.front()->get();

    auto outsDefiningOp = outputOperand.getDefiningOp<linalg::LinalgOp>();
    if (!outsDefiningOp || isa<linalg::FillOp>(outsDefiningOp.getOperation())) {
      // If not linalg op, or is a fill op, do nothing.
      return failure();
    }
    auto outputType = llvm::cast<RankedTensorType>(outputOperand.getType());
    if (!outputType.getElementType().isIntOrFloat())
      return failure();
    auto elementType = outputType.getElementType();

    Location loc = linalgOp.getLoc();

    // Create a zero tensor as the new output tensor operand to the Linalg
    // contraction op.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(rewriter, loc, outputOperand);
    auto initOp =
        rewriter.create<tensor::EmptyOp>(loc, mixedSizes, elementType);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
    Value fill =
        rewriter.create<linalg::FillOp>(loc, zero, initOp.getResult()).result();

    // Update the contraction op to use the new zero tensor as output operand.
    rewriter.updateRootInPlace(linalgOp,
                               [&]() { linalgOp.setDpsInitOperand(0, fill); });

    auto outputMap = mlir::compressUnusedDims(
        linalgOp.getMatchingIndexingMap(outputOperands.front()));
    // Only support identity map for output access for now; this is the case for
    // all existing contraction/convolution ops.
    if (!outputMap.isIdentity())
      return failure();
    SmallVector<AffineMap> maps(3, outputMap);

    SmallVector<utils::IteratorType> iterators;
    iterators.reserve(outputMap.getNumResults());
    for (int i = 0, e = outputMap.getNumResults(); i < e; ++i) {
      int pos = outputMap.getResult(i).cast<AffineDimExpr>().getPosition();
      auto attr = linalgOp.getIteratorTypesArray()[pos];
      if (!linalg::isParallelIterator(attr))
        return failure();
      iterators.push_back(attr);
    }

    // Create a generic op to add back the original output tensor operand.
    rewriter.setInsertionPointAfter(linalgOp);
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{linalgOp->getResult(0), outputOperand},
        fill, maps, iterators,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result;
          if (llvm::isa<FloatType>(elementType)) {
            result = b.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
          } else {
            result = b.create<arith::AddIOp>(nestedLoc, args[0], args[1]);
          }
          b.create<linalg::YieldOp>(nestedLoc, result);
        });
    linalgOp->getResult(0).replaceAllUsesExcept(genericOp->getResult(0),
                                                genericOp);
    return success();
  }
};

template <typename InterfaceOp>
struct DetachSplatConstantOutsOperands
    : public OpInterfaceRewritePattern<InterfaceOp> {
  using OpInterfaceRewritePattern<InterfaceOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(InterfaceOp interfaceOp,
                                PatternRewriter &rewriter) const {
    SmallVector<Value> newOutsOperands;
    auto dpsInterfaceOp =
        dyn_cast<DestinationStyleOpInterface>(interfaceOp.getOperation());
    if (!dpsInterfaceOp) {
      return rewriter.notifyMatchFailure(
          interfaceOp, "expected op to implement DPS interface");
    }
    bool madeChanges = false;
    for (auto outOperand :
         llvm::enumerate(dpsInterfaceOp.getDpsInitOperands())) {
      auto constOp =
          outOperand.value()->get().template getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        continue;

      auto resultType =
          llvm::dyn_cast<RankedTensorType>(constOp.getResult().getType());
      if (!resultType || !resultType.getElementType().isIntOrFloat())
        continue;

      auto attr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!attr || !attr.isSplat())
        continue;

      Location loc = constOp.getLoc();
      Type elementType = resultType.getElementType();
      Value emptyTensorOp = rewriter.create<tensor::EmptyOp>(
          loc, resultType.getShape(), elementType);
      TypedAttr constValue;
      if (llvm::isa<IntegerType>(elementType)) {
        constValue = rewriter.getIntegerAttr(
            elementType, attr.template getSplatValue<APInt>());
      } else {
        constValue = rewriter.getFloatAttr(
            elementType, attr.template getSplatValue<APFloat>());
      }
      Value scalarConstantOp =
          rewriter.create<arith::ConstantOp>(loc, elementType, constValue);

      Value fillOp = rewriter
                         .create<linalg::FillOp>(
                             loc, resultType, scalarConstantOp, emptyTensorOp)
                         .getResult(0);
      rewriter.updateRootInPlace(dpsInterfaceOp, [&]() {
        dpsInterfaceOp.setDpsInitOperand(outOperand.index(), fillOp);
      });
      madeChanges = true;
    }
    return success(madeChanges);
  };
};

struct DetachElementwiseFromNamedOpsPass
    : public DetachElementwiseFromNamedOpsBase<
          DetachElementwiseFromNamedOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DetachElementwisePattern,
                 DetachSplatConstantOutsOperands<linalg::LinalgOp>>(
        &getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass() {
  return std::make_unique<DetachElementwiseFromNamedOpsPass>();
}
} // namespace  mlir
