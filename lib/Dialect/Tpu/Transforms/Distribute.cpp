//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/Distribute/Distribute.h"

using namespace llvm;
namespace tpu_mlir {
namespace tpu {
// only >= 4MB distribute to multi devices
static const int64_t WEIGHT_LIMIT = 0x400000;

void distribute(PatternRewriter &rewriter, Operation *op_begin,
                Operation *op_end, tpu::DistributionPattern pattern) {
  auto ctx = rewriter.getContext();
  auto input = op_begin->getOperand(0);
  std::vector<Type> types = {input.getType()};
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::DistributionPatternAttr::get(ctx, pattern)));
  auto begin_loc = module::getLocLike(op_begin, "begin");
  rewriter.setInsertionPoint(op_begin);
  auto begin = rewriter.create<tpu::DistributionBeginOp>(
      begin_loc, types, ValueRange{input}, attrs);
  op_begin->setOperand(0, begin.getOutput());

  Value output;
  for (auto o : op_end->getResults()) {
    if (o.hasOneUse()) {
      output = o;
      break;
    }
  }
  auto end_loc = module::getLocLike(output, "end");
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DistributionEndOp>(end_loc, output.getType(),
                                                     ValueRange{output}, attrs);
  output.replaceAllUsesExcept(end.getOutput(), end);
}

bool isLargeMatMul(Operation *op) {
  auto mm = dyn_cast<tpu::MatMulOp>(op);
  if (!mm) {
    return false;
  }
  if (module::isWeight(mm.getRight()) == false) {
    return false;
  }
  auto num_right = module::getNumElements(mm.getRight());
  if (num_right <= WEIGHT_LIMIT) {
    return false;
  }
  return true;
}

bool isBinaryOp(Operation *op) { return isa<tpu::AddOp, tpu::MulOp>(op); }

Operation *cloneOp(PatternRewriter &rewriter, Operation *op,
                   llvm::ArrayRef<int64_t> new_shape, llvm::StringRef suffix) {
  rewriter.setInsertionPointAfter(op);
  auto new_op = rewriter.clone(*op);
  for (auto r : new_op->getResults()) {
    module::setShape(r, new_shape);
  }
  module::setLocSuffix(new_op, suffix);
  return new_op;
}

void eraseForward(PatternRewriter &rewriter, Operation *op) {
  if (!op->use_empty()) {
    for (auto u : op->getUsers()) {
      eraseForward(rewriter, u);
    }
  }
  rewriter.eraseOp(op);
}

// ===================================
// distribute all ops to multi device
// ===================================
class DoDistributePattern : public OpRewritePattern<tpu::DistributionBeginOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  DoDistributePattern(MLIRContext *context)
      : OpRewritePattern<tpu::DistributionBeginOp>(context) {
    num_devices = module::getDeviceNum();
  }
  LogicalResult matchAndRewrite(tpu::DistributionBeginOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDone()) {
      return failure();
    }
    switch (op.getPattern()) {
    case tpu::DistributionPattern::MatMulSliceMerge:
      DoDistribution<MatMulSliceMerge>(rewriter, op, num_devices);
      break;
    case tpu::DistributionPattern::MatMulTopK:
      DoDistribution<MatMulTopK>(rewriter, op, num_devices);
      break;
    default:
      return failure();
    }
    op.setDone(true);
    return success();
  }

private:
  int64_t num_devices;
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
    mOp = getOperation();
    ctx = &getContext();
    applyPattern<MatMulSliceMerge>(mOp);
    applyPattern<MatMulTopK>(mOp);
    applyPattern<DoDistributePattern>(mOp);
    DistributeModules(mOp, num_device);
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
    // OpBuilder build0id0(module0id0.getBody(),
    // module0id0.getBody()->begin());
  }

private:
  mlir::MLIRContext *ctx;
  ModuleOp mOp;
};

std::unique_ptr<OperationPass<ModuleOp>> createDistributePass() {
  return std::make_unique<DistributePass>();
}
} // namespace tpu
} // namespace tpu_mlir
