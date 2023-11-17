//===- TestTilingInterface.cpp - Test tiling using `TilingInterface` -----===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing tiling operations using
// `TilingInterface`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {

/// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
struct TilingOpFilter {

  explicit TilingOpFilter(ArrayRef<StringAttr> matchDisjunction = {},
                          std::optional<StringAttr> replacement = std::nullopt);

  TilingOpFilter(TilingOpFilter &&) = default;
  TilingOpFilter(const TilingOpFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;

private:
  SmallVector<StringAttr> matchDisjunction;
  std::optional<StringAttr> replacement;
};

TilingOpFilter::TilingOpFilter(ArrayRef<StringAttr> matchDisjunction,
                               std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement) {}

LogicalResult TilingOpFilter::checkAndNotify(PatternRewriter &rewriter,
                                             Operation *op) const {

  auto attr = op->template getAttrOfType<StringAttr>(kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty())
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void TilingOpFilter::replaceLinalgTransformationFilter(
    PatternRewriter &rewriter, Operation *op) const {
  if (replacement.has_value())
    op->setAttr(kLinalgTransformMarker, *replacement);
  else
    op->removeAttr(rewriter.getStringAttr(kLinalgTransformMarker));
}

/// Pattern for testing `TileConsumerAndFuseProducersUsingSCFForOp` pattern
/// (that tiles and fuses operations using the `TilingInterface` with `scf.for`
/// ops for iterating over the tiles) while using a `filter` to avoid recursive
/// application.
struct TestTileConsumerAndFuseProducersGreedilyUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {
  TestTileConsumerAndFuseProducersGreedilyUsingSCFForOp(
      MLIRContext *context, scf::SCFTileAndFuseOptions options,
      TilingOpFilter filter = TilingOpFilter(), PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        options(std::move(options)), filter(std::move(filter)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(rewriter, op,
                                                              options);
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

    filter.replaceLinalgTransformationFilter(
        rewriter, tileAndFuseResult->tiledAndFusedOps.front());
    return success();
  }

private:
  scf::SCFTileAndFuseOptions options;
  TilingOpFilter filter;
};

/// Test pass for testing the use of `TilingInterface`.
struct TestTilingInterfacePass
    : public PassWrapper<TestTilingInterfacePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTilingInterfacePass)

  TestTilingInterfacePass() = default;
  TestTilingInterfacePass(const TestTilingInterfacePass &pass)
      : PassWrapper(pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
  }
  StringRef getArgument() const final { return "test-tiling-interface"; }
  StringRef getDescription() const final {
    return "Test tiling using TilingInterface";
  }

  ListOption<int> testTileConsumerAndFuseProducer{
      *this, "tile-size",
      llvm::cl::desc("Test tile and fuse transformation using TilingInterface "
                     "with scf.for operations")};

  void runOnOperation() override;

private:
  void addTestPatterns(MLIRContext *context, RewritePatternSet &patterns);
};
} // namespace

static void addPatternForTileAndFuse(MLIRContext *context,
                                     RewritePatternSet &patterns,
                                     StringRef filterName,
                                     ArrayRef<int64_t> tileSizes,
                                     ArrayRef<int64_t> interchange = {}) {
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.tilingOptions.setTileSizes(tileSizes).setInterchange(
      interchange);
  TilingOpFilter filter(StringAttr::get(context, filterName),
                        StringAttr::get(context, "tiled"));
  patterns.add<TestTileConsumerAndFuseProducersGreedilyUsingSCFForOp>(
      context, tileAndFuseOptions, filter);
}

void TestTilingInterfacePass::addTestPatterns(MLIRContext *context,
                                              RewritePatternSet &patterns) {
  if (testTileConsumerAndFuseProducer.hasValue()) {
    SmallVector<int64_t> tileSizes(*testTileConsumerAndFuseProducer);
    addPatternForTileAndFuse(context, patterns, "root", tileSizes);
    return;
  }
}

void TestTilingInterfacePass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet tilingPatterns(context);
  addTestPatterns(context, tilingPatterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(tilingPatterns))))
    return signalPassFailure();
}

namespace tpu_mlir {
namespace test {
void registerTestTilingInterface() {
  PassRegistration<TestTilingInterfacePass>();
}
} // namespace test
} // namespace tpu_mlir
