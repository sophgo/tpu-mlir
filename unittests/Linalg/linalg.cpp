//===-- Linalg.cpp - linalg feature explore -------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "TPUDialect.h"
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
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

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

class RemoveRedudentCopy : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (isa_and_nonnull<memref::AllocOp>(copyOp.getTarget().getDefiningOp())) {
      if (isa_and_nonnull<memref::AllocOp>(
              copyOp.getSource().getDefiningOp())) {
        rewriter.replaceAllUsesWith(copyOp.getTarget(), copyOp.getSource());
        rewriter.eraseOp(copyOp);
        return success();
      } else if (auto args = dyn_cast<BlockArgument>(copyOp.getSource())) {
        if (isa<scf::ForOp>(args.getOwner()->getParentOp())) {
          rewriter.replaceAllUsesWith(copyOp.getTarget(), copyOp.getSource());
          rewriter.eraseOp(copyOp);
          return success();
        }
        return failure();
      }
      return failure();
    }
    return failure();
  };
};

struct EmptyTensorLoweringPattern : public OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern<tensor::EmptyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::EmptyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<bufferization::AllocTensorOp>(
        op, op.getType(), op.getDynamicSizes());
    return success();
  }
};

class AllocLocalMem : public OpRewritePattern<memref::AllocOp> {
public:
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!isa_and_nonnull<scf::ForOp>(allocOp->getParentOp()))
      return failure();
    auto type = allocOp.getType();
    if (type.getLayout().isIdentity()) {
      auto ctx = getContext();
      AffineExpr d0, d1, d2, d3, s0;
      bindDims(ctx, d0, d1, d2, d3);
      bindSymbols(ctx, s0);
      auto c1 = mlir::getAffineConstantExpr(1, ctx);
      AffineMap mm2Layout = AffineMap::get(2, 0, {c1, d0, c1, d1}, ctx);
      AffineMap mm2Con =
          AffineMap::get(4, 1, {d0, d1.ceilDiv(s0), s0, d2, d3}, ctx);
      NamedAttrList attrs;
      attrs.set("address",
                rewriter.getIntegerAttr(rewriter.getI64Type(), 2260));
      attrs.set("shape", AffineMapAttr::get(mm2Con));
      attrs.set("mem_type", rewriter.getStringAttr("local memory"));
      auto newType =
          MemRefType::get(type.getShape(), type.getElementType(), mm2Layout,
                          rewriter.getDictionaryAttr(attrs));
      allocOp.getResult().setType(newType);
      return success();
    }
    return failure();
  };
};

class MemView2TPUView : public OpRewritePattern<memref::SubViewOp> {
public:
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<int64_t> offsets;
    for (auto s : op.getOffsets()) {
      if (auto ct = dyn_cast<arith::ConstantOp>(s.getDefiningOp()))
        if (auto v = getConstantIntValue(s))
          offsets.push_back(v.value());
    }
    if (!offsets.empty()) {
      auto subSizes = op.getStaticSizes();
      auto subStrides = op.getStaticStrides();
      rewriter.replaceOpWithNewOp<tpu::SubViewOp>(
          op, op.getType(), op.getSource(), subSizes, offsets, subStrides);

      return success();
    }
    return failure();
  };
};

class MaterializeMemViewLayout : public OpRewritePattern<tpu::SubViewOp> {
public:
  using OpRewritePattern<tpu::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SubViewOp op,
                                PatternRewriter &rewriter) const override {

    auto outT = op.getType().cast<MemRefType>();
    if (auto strided = llvm::dyn_cast<StridedLayoutAttr>(outT.getLayout())) {
      auto offset = linearize(op.getStaticOffsets(), strided.getStrides());
      auto ctx = getContext();
      AffineExpr d0, d1, d2, d3, s0;
      bindDims(ctx, d0, d1, d2, d3);
      bindSymbols(ctx, s0);
      auto c1 = mlir::getAffineConstantExpr(1, ctx);
      auto stride = strided.getStrides();
      AffineMap mm2Layout = AffineMap::get(2, 0, {c1, d0, c1, d1}, ctx);
      auto satt = rewriter.getI64ArrayAttr({0, stride[0], 0, stride[1]});
      NamedAttrList attrs;
      attrs.set("offset",
                rewriter.getIntegerAttr(rewriter.getI64Type(), offset));
      attrs.set("stride", satt);
      attrs.set("mem_type", rewriter.getStringAttr("global memory"));
      auto newType =
          MemRefType::get(outT.getShape(), outT.getElementType(), mm2Layout,
                          rewriter.getDictionaryAttr(attrs));
      op.getResult().setType(newType);
      return success();
    }

    return failure();
  };
};

class ApplyLayoutT : public OpRewritePattern<tpu::SubViewOp> {
public:
  using OpRewritePattern<tpu::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::SubViewOp op,
                                PatternRewriter &rewriter) const override {

    auto outT = op.getType().cast<MemRefType>();
    if (!outT.getLayout().isIdentity()) {
      // apply shape transform
      auto shape = outT.getShape();
      auto newType =
          MemRefType::get({1, shape[0], 1, shape[1]}, outT.getElementType(),
                          AffineMap::getMultiDimIdentityMap(4, getContext()),
                          outT.getMemorySpace());
      op.getResult().setType(newType);
      return success();
    }

    return failure();
  };
};

class ApplyLayoutM : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    auto outT = op.getTarget().getType().cast<MemRefType>();
    if (!outT.getLayout().isIdentity()) {
      // apply shape transform
      auto shape = outT.getShape();
      auto newType =
          MemRefType::get({1, shape[0], 1, shape[1]}, outT.getElementType(),
                          AffineMap::getMultiDimIdentityMap(4, op.getContext()),
                          outT.getMemorySpace());
      op.getTarget().setType(newType);
      return success();
    }

    return failure();
  };
};

class Linalg2TPU : public OpRewritePattern<linalg::MatmulOp> {
public:
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<tpu::MatMulOp>(
        op, op.getInputs()[0], op.getInputs()[0], op.getOutputs()[0]);
    return success();
  };
};

struct SimplifyPass0
    : public PassWrapper<SimplifyPass0, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyPass0)

  SimplifyPass0() = default;
  SimplifyPass0(const SimplifyPass0 &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return "simplify0"; }
  StringRef getDescription() const final {
    return "Simply memref alloc and copy";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet simpilyPatterns(context);
    simpilyPatterns.add<EmptyTensorLoweringPattern>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(simpilyPatterns))))
      return signalPassFailure();
  };
};

struct SimplifyPass
    : public PassWrapper<SimplifyPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyPass)

  SimplifyPass() = default;
  SimplifyPass(const SimplifyPass &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return "simplify"; }
  StringRef getDescription() const final {
    return "Simply memref alloc and copy";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet simpilyPatterns(context);
    simpilyPatterns.add<RemoveRedudentCopy>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(simpilyPatterns))))
      return signalPassFailure();
  };
};

struct LocalMemInSCFPass
    : public PassWrapper<LocalMemInSCFPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LocalMemInSCFPass)

  LocalMemInSCFPass() = default;
  LocalMemInSCFPass(const LocalMemInSCFPass &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return ""; }
  StringRef getDescription() const final {
    return "Annotate local memory layout.";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet allocPatterns(context);
    allocPatterns.add<AllocLocalMem>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(allocPatterns))))
      return signalPassFailure();
  };
};

struct ApplyStaticInfo
    : public PassWrapper<ApplyStaticInfo, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplyStaticInfo)

  ApplyStaticInfo() = default;
  ApplyStaticInfo(const ApplyStaticInfo &pass) : PassWrapper(pass) {}
  StringRef getArgument() const final { return ""; }
  StringRef getDescription() const final { return "Apply static information."; }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet allocPatterns(context);
    allocPatterns.add<Linalg2TPU>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(allocPatterns))))
      return signalPassFailure();

    allocPatterns.clear();
    allocPatterns.add<MemView2TPUView, MaterializeMemViewLayout, ApplyLayoutM,
                      ApplyLayoutT>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(allocPatterns))))
      return signalPassFailure();
  };
};

static unsigned getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<scf::ForOp>(currOp))
      depth++;
  }
  return depth;
}

namespace {
TEST(LinalgGenericOp, TileMatMul) {
  using namespace linalg;

  DialectRegistry registry;
  registry.insert<LinalgDialect, func::FuncDialect, affine::AffineDialect,
                  scf::SCFDialect, tensor::TensorDialect, tpu::TPUDialect>();

  linalg::registerTilingInterfaceExternalModels(registry);

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  Builder builder(&context);

  std::string moduleStr = R"mlir(
func.func @outerproduct_matmul(%A: tensor<512x512xf16>, %B: tensor<512x512xf16>) -> tensor<512x512xf16> {
  %cst = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<512x512xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<512x512xf16>) -> tensor<512x512xf16>
  %d = linalg.matmul ins(%A, %B: tensor<512x512xf16>, tensor<512x512xf16>)
            outs(%fill: tensor<512x512xf16>) -> tensor<512x512xf16>
  return %d : tensor<512x512xf16>
}
                           )mlir";

  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  auto funcOp = cast<func::FuncOp>(*module->getBodyRegion().getOps().begin());
  for (auto matmul : funcOp.getOps<MatmulOp>()) {

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes({64, 64, 128});

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
  // module->dump();
  // extract tensor need to assign buffer
  // 1. use local layer, except one have global layer only
  // 2. SCF have captured all the tensor used by outside, those should be
  // allocated in global memory.
  //   a. Tensors in SCF region do not need to assign memory in global memory.
  //   b. the results of SCF operation are the value used by outside
  // 3. convert parts of tensor.extract_slice to tensor.transfer and those will
  // be lowering to DMA operation(or to memref.view, memref.view can not encode
  // all the information need by CodeGen, we need to inset dma.transfer to
  // convert the data in DDR to local memory with a different layout).
  //   a. reduction should in the inner loop to provide better
  //   b. parallel dim should be in the outer loop to help the distribution
  // convert buffer to memmref decompose the
  // operation, and it can be select(lowering) by tpu-kernel assign address to
  // memref codegen
  SmallVector<Value> tensorToMemref;
  module->walk<WalkOrder::PreOrder>([&tensorToMemref](LinalgOp op) {
    if (isa<linalg::FillOp>(op)) {
      return WalkResult::skip();
    }
    tensorToMemref.append(op->getOperands().begin(), op->getOperands().end());
    return WalkResult::advance();
  });
  // find the value used in SCF region
  for (auto v : tensorToMemref) {
    v.getType().dump();
  }

  pm.clear();
  pm.addPass(createCSEPass());

  pm.addNestedPass<func::FuncOp>(createSCFBufferizePass());
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.addNestedPass<func::FuncOp>(mlir::tensor::createTensorBufferizePass());
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createBufferizationBufferizePass());
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  pm.addNestedPass<func::FuncOp>(std::make_unique<SimplifyPass>());
  pm.nest<func::FuncOp>().addPass(std::make_unique<LocalMemInSCFPass>());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.run(module->getOperation());

  // module->dump();

  // unroll loop
  SmallVector<scf::ForOp, 4> loops;
  SmallVector<int, 4> unrollFactors{4, 8, 8};
  funcOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (auto [loop, factor] : llvm::zip(loops, unrollFactors))
    loopUnrollByFactor(loop, factor);

  pm.clear();
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.run(module->getOperation());
  // module->dump();

  // apply static information
  pm.clear();
  pm.addNestedPass<func::FuncOp>(std::make_unique<ApplyStaticInfo>());
  // pm.addPass(createCanonicalizerPass());
  // pm.addPass(createCSEPass());

  pm.run(module->getOperation());
  // module->dump();
  // optimize memory allocate
}

/// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";

/// Helper class to control application of linalg transformation patterns.
/// Control comes in 2 forms:
///   1. attribute matching and setting behavior using the attribute named
///      `kLinalgTransformMarker`. This can be used to build a state machine
///      using attributes and incrementally applying patterns to advance states.
///   2. filter function, which is a simple lambda on the Operation* that
///      returns a LogicalResult.
/// 辅助类，用于控制线性代数变换模式的应用。
/// 控制有两种形式：
///   1. 使用名为`kLinalgTransformMarker`的属性进行属性匹配和设置行为。可以使用属性构建状态机，并逐步应用模式以推进状态。
///   2. 过滤函数，它是对Operation*的简单lambda表达式，返回LogicalResult。
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
  llvm::errs() << "matchDisjunction:\n";
  for (auto i: matchDisjunction)
    llvm::errs() << i.str()<< "\n";
  llvm::errs() << "replacement:" << replacement->str()<< "\n";
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

TEST(LinalgGenericOp, TileConv2d) {
  using namespace linalg;

  DialectRegistry registry;
  registry.insert<LinalgDialect, func::FuncDialect, affine::AffineDialect,
                  scf::SCFDialect, tensor::TensorDialect, tpu::TPUDialect>();

  linalg::registerTilingInterfaceExternalModels(registry);

  MLIRContext context(registry);

  for (StringRef name : registry.getDialectNames())
    context.getOrLoadDialect(name);

  Builder builder(&context);

  std::string moduleStr = R"mlir(
func.func @conv_tensors_static(%input: tensor<4x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>, %filter_0: tensor<3x3x32x32xf32>) -> tensor<4x110x110x32xf32> {
  %init = tensor.empty() : tensor<4x112x112x32xf32>
  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter : tensor<4x225x225x3xf32>, tensor<3x3x3x32xf32>)
    outs(%init : tensor<4x112x112x32xf32>) -> tensor<4x112x112x32xf32>

  %init_0 = tensor.empty() : tensor<4x110x110x32xf32>
  %conv_0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>,
    __internal_linalg_transform__ = "root"}
    ins(%conv, %filter_0 : tensor<4x112x112x32xf32>, tensor<3x3x32x32xf32>)
    outs(%init_0 : tensor<4x110x110x32xf32>) -> tensor<4x110x110x32xf32>
  return %conv_0: tensor<4x110x110x32xf32>
}
  )mlir";

  TilingOpFilter filter(mlir::StringAttr::get(&context, "root"),
                        mlir::StringAttr::get(&context, "tiled"));

  auto module = parseSourceString<ModuleOp>(moduleStr, &context);
  auto funcOp = cast<func::FuncOp>(*module->getBodyRegion().getOps().begin());
  //for (auto linalgOp : funcOp.getOps<Conv2DNhwcHwcfOp>()) {
  for (auto linalgOp : funcOp.getOps<LinalgOp>()) {
    class MyPatternRewriter : public PatternRewriter {
    public:
      MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
    };
    MyPatternRewriter rewriter(&context);
    auto tileif = dyn_cast<TilingInterface>(linalgOp.getOperation());
    if (failed(filter.checkAndNotify(rewriter, tileif))) {
      printf("checkAndNotify fail\n");
      continue;
    }
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.tilingOptions.setTileSizes({2, 16, 0, 0}); //.setInterchange(interchange);

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult = scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
        rewriter, tileif,
        tileAndFuseOptions);
    if (failed(tileAndFuseResult)) {
      printf("tileAndFuseResult fail\n");
      continue;
    }

    SmallVector<Value> replacements(linalgOp->getNumResults());
    for (const auto &result : llvm::enumerate(linalgOp->getResults())) {
      replacements[result.index()] =
          tileAndFuseResult->replacements.lookup(result.value());
    }
    rewriter.replaceOp(tileif, replacements);
    filter.replaceLinalgTransformationFilter(
        rewriter, tileAndFuseResult->tiledAndFusedOps.front());
  }

  auto pm = PassManager::on<mlir::ModuleOp>(&context);
  pm.addPass(createCSEPass());
  pm.run(module->getOperation());
  printf("mlirfile after tile-and-fuse\n");
  module->dump();
  // extract tensor need to assign buffer
  // 1. use local layer, except one have global layer only
  // 2. SCF have captured all the tensor used by outside, those should be
  // allocated in global memory.
  //   a. Tensors in SCF region do not need to assign memory in global memory.
  //   b. the results of SCF operation are the value used by outside
  // 3. convert parts of tensor.extract_slice to tensor.transfer and those will
  // be lowering to DMA operation(or to memref.view, memref.view can not encode
  // all the information need by CodeGen, we need to inset dma.transfer to
  // convert the data in DDR to local memory with a different layout).
  //   a. reduction should in the inner loop to provide better
  //   b. parallel dim should be in the outer loop to help the distribution
  // convert buffer to memmref decompose the
  // operation, and it can be select(lowering) by tpu-kernel assign address to
  // memref codegen

  // SmallVector<Value> tensorToMemref;
  // module->walk<WalkOrder::PreOrder>([&tensorToMemref](LinalgOp op) {
  //   if (isa<linalg::FillOp>(op)) {
  //     return WalkResult::skip();
  //   }
  //   tensorToMemref.append(op->getOperands().begin(), op->getOperands().end());
  //   return WalkResult::advance();
  // });
  // // find the value used in SCF region
  // for (auto v : tensorToMemref) {
  //   v.getType().dump();
  // }

  pm.clear();
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(std::make_unique<SimplifyPass0>());
  pm.addNestedPass<func::FuncOp>(createSCFBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createSCFBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createLinalgBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addNestedPass<func::FuncOp>(mlir::tensor::createTensorBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createTensorBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addPass(mlir::func::createFuncBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createFuncBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createBufferizationBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createBufferizationBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());
  pm.run(module->getOperation());
  printf("mlirfile after createFinalizingBufferizePass\n");
  module->dump();

  pm.clear();
  pm.addNestedPass<func::FuncOp>(std::make_unique<SimplifyPass>());
  // pm.nest<func::FuncOp>().addPass(std::make_unique<LocalMemInSCFPass>());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.run(module->getOperation());
  printf("mlirfile after SimplifyPass and createCanonicalizerPass\n");
  module->dump();

  // unroll loop
  SmallVector<scf::ForOp, 2> loops;
  funcOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  SmallVector<int, 2> unrollFactors{4,10};
  for (auto [loop, factor] : llvm::zip(loops, unrollFactors))
    loopUnrollByFactor(loop, factor);

  pm.clear();
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.run(module->getOperation());
  printf("mlirfile after unroll loop\n");
  module->dump();

  // // apply static information
  // pm.clear();
  // pm.addNestedPass<func::FuncOp>(std::make_unique<ApplyStaticInfo>());
  // // pm.addPass(createCanonicalizerPass());
  // // pm.addPass(createCSEPass());
  // pm.run(module->getOperation());
  // printf("mlirfile after ApplyStaticInfo\n");
  // module->dump();
  // // optimize memory allocate
}

} // namespace
