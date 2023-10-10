//===-- Linalg.cpp - linalg feature explore -------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/FormatVariadic.h"
#include "gtest/gtest.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;
using namespace mlir::detail;

struct TestTileUsingSCFForOp
    : public OpInterfaceRewritePattern<TilingInterface> {
  TestTileUsingSCFForOp(MLIRContext *context, scf::SCFTilingOptions options,
                        PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCFForOp(rewriter, op, options);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(op, "failed to tile operation");
    return success();
  }

private:
  scf::SCFTilingOptions options;
};

namespace {
TEST(LinalgGenericOp, TileMatMul) {
  using namespace linalg;

  DialectRegistry registry;
  registry.insert<LinalgDialect, func::FuncDialect, affine::AffineDialect,
                  scf::SCFDialect, tensor::TensorDialect>();

  linalg::registerTilingInterfaceExternalModels(registry);

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  Builder builder(&context);

  std::string moduleStr = R"mlir(
func.func @outerproduct_matmul(%A: tensor<300x300xf32>, %B: tensor<300x300xf32>) -> tensor<300x300xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<300x300xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<300x300xf32>) -> tensor<300x300xf32>
  %d = linalg.matmul ins(%A, %B: tensor<300x300xf32>, tensor<300x300xf32>)
            outs(%fill: tensor<300x300xf32>) -> tensor<300x300xf32>
  return %d : tensor<300x300xf32>
}
                           )mlir";

  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  auto funcOp = cast<func::FuncOp>(*module->getBodyRegion().getOps().begin());
  for (auto matmul : funcOp.getOps<MatmulOp>()) {

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes({32, 8, 32});

    class MyPatternRewriter : public PatternRewriter {
    public:
      MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
    };

    MyPatternRewriter rewriter(&context);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCFForOp(
        rewriter, dyn_cast<TilingInterface>(matmul.getOperation()),
        tilingOptions);

    rewriter.replaceOp(matmul, tilingResult->replacements);
  }

  auto pm = PassManager::on<mlir::ModuleOp>(&context);
  pm.addPass(createCSEPass());
  pm.run(module->getOperation());
  module->dump();
}

} // namespace
