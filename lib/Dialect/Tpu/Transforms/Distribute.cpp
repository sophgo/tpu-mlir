//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Support/Debug.h>

using namespace llvm;

namespace tpu_mlir {
namespace tpu {
// only >= 4MB distribute to multi devices
static const int64_t WEIGHT_LIMIT = 0x400000;

static bool isOpInDistribution(Operation *op) {
  auto user = op;
  while (user != nullptr && !isa<func::ReturnOp>(user)) {
    if (isa<tpu::ConnectOp>(user)) {
      return true;
    }
    if (user->hasOneUse() == false) {
      return false;
    }
    user = *user->getUsers().begin();
  }
  return false;
}

static void distribute(PatternRewriter &rewriter, Operation *op,
                       bool need_slice, tpu::ConnectMode mode) {
  auto name = module::getName(op).str();
  auto ctx = rewriter.getContext();
  auto input = op->getOperand(0);
  auto output = op->getResult(0);
  std::vector<Type> types = {input.getType()};
  std::vector<NamedAttribute> distribute_attrs;
  distribute_attrs.push_back(
      rewriter.getNamedAttr("do_slice", rewriter.getBoolAttr(need_slice)));
  auto distribute_loc =
      NameLoc::get(rewriter.getStringAttr(name + "_distribute"));
  rewriter.setInsertionPoint(op);
  auto distribute = rewriter.create<tpu::DistributeOp>(
      distribute_loc, types, ValueRange{input}, distribute_attrs);
  op->setOperand(0, distribute.getResult(0));

  std::vector<NamedAttribute> connect_attrs;
  connect_attrs.push_back(rewriter.getNamedAttr(
      "mode", tpu::ConnectModeAttr::get(ctx, tpu::ConnectMode::Concat)));
  auto connect_loc = NameLoc::get(rewriter.getStringAttr(name + "_connect"));
  rewriter.setInsertionPointAfter(op);
  auto connect = rewriter.create<tpu::ConnectOp>(
      connect_loc, output.getType(), ValueRange{output}, connect_attrs);
  output.replaceAllUsesExcept(connect.getOutput(), connect);
}

class MoveConnectPattern : public OpRewritePattern<tpu::ConnectOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ConnectOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

class MatMulDistributePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isWeight(op.getRight()) == false) {
      return failure();
    }
    auto out_stype = module::getStorageType(op.getOutput());
    if (!out_stype.isa<FloatType>()) {
      return failure();
    }
    auto num_right = module::getNumElements(op.getRight());
    if (num_right <= WEIGHT_LIMIT) {
      return failure();
    }
    if (isOpInDistribution(op)) {
      return failure();
    }
    auto shape = module::getShape(op.getRight());
    auto K = shape[shape.size() - 2];
    auto N = shape[shape.size() - 1];
    auto mode = tpu::ConnectMode::Add;
    bool need_slice = true;
    if (K <= N || op.getDoRelu()) {
      mode = tpu::ConnectMode::Concat;
      need_slice = false;
    }
    distribute(rewriter, op, need_slice, mode);
    return success();
  }
};

class DistributePass : public DistributeBase<DistributePass> {
public:
  DistributePass() {}
  void runOnOperation() override {
    if (num_device <= 1) {
      return;
    }
    if (module::isBM1684XFamily() == false) {
      return;
    }
    module::setDeviceNum(num_device);
    auto mOp = getOperation();
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<MatMulDistributePattern>(ctx);
    auto config = GreedyRewriteConfig();
    config.maxIterations = 1; // apply each pattern only once.
    applyPatternsAndFoldGreedily(mOp, std::move(patterns), config);

    // OpBuilder builder(&getContext());
    // auto oriModule = getOperation();
    // auto attrs = oriModule->getAttrs();
    // StringRef name0 = "device0";
    // builder.setInsertionPointToEnd(oriModule.getBody());
    // auto loc0 = NameLoc::get(builder.getStringAttr(name0));
    // auto module0 = builder.create<ModuleOp>(loc0);
    // OpBuilder build0(module0.getBody(), module0.getBody()->begin());
    // StringRef id0 = "id0";
    // auto loc0id0 = NameLoc::get(builder.getStringAttr(id0));
    // auto module0id0 = build0.create<ModuleOp>(loc0id0);
    // module0id0->setAttrs(attrs);
    // OpBuilder build0id0(module0id0.getBody(), module0id0.getBody()->begin());
  }

private:
};

std::unique_ptr<OperationPass<ModuleOp>> createDistributePass() {
  return std::make_unique<DistributePass>();
}
} // namespace tpu
} // namespace tpu_mlir
