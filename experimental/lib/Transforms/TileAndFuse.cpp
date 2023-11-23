//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu-mlir/Transforms/Passes.h"

namespace tpu_mlir {
using namespace mlir;

const StringLiteral kLinalgTilingMarker = "__linalg_tiling__";

struct TileConsumerAndFuseProducersGreedilyUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {
  TileConsumerAndFuseProducersGreedilyUsingSCFForOp(MLIRContext *context,
                                                    PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    auto attr =
        op->template getAttrOfType<DenseI64ArrayAttr>(kLinalgTilingMarker);

    if (!attr) {
      if (op->hasAttr(kLinalgTilingMarker)) {
        return op->emitOpError(
            kLinalgTilingMarker +
            " needs array<i64: .+> type. But got unknown parameter");
      }
      return failure();
    }

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.tilingOptions.setTileSizes(attr.asArrayRef());

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
            rewriter, op, tileAndFuseOptions);

    if (failed(tileAndFuseResult)) {
      return failure();
    }
    // Replace the tiled op with replacements.
    SmallVector<Value> replacements(op->getNumResults());
    for (const auto &result : llvm::enumerate(op->getResults())) {
      replacements[result.index()] =
          tileAndFuseResult->replacements.lookup(result.value());
    }
    rewriter.replaceOp(op, replacements);

    tileAndFuseResult->tiledAndFusedOps.front()->removeAttr(
        rewriter.getStringAttr(kLinalgTilingMarker));

    return success();
  }
};

class TileAndFuseGreedily
    : public TileAndFuseGreedilyBase<TileAndFuseGreedily> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet tilingPatterns(context);
    tilingPatterns.add<TileConsumerAndFuseProducersGreedilyUsingSCFForOp>(
        context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(tilingPatterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<Pass> createTileAndFuseGreedilyPass() {
  return std::make_unique<TileAndFuseGreedily>();
}

} // namespace tpu_mlir
