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
  rewriter.setInsertionPoint(op_begin);
  auto begin = rewriter.create<tpu::DistributionBeginOp>(
      op_begin->getLoc(), types, ValueRange{input}, attrs);
  op_begin->setOperand(0, begin.getOutput());

  Value output;
  for (auto o : op_end->getResults()) {
    if (o.hasOneUse()) {
      output = o;
      break;
    }
  }
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DistributionEndOp>(
      module::getLoc(output), output.getType(), ValueRange{output}, attrs);
  output.replaceAllUsesExcept(end.getOutput(), end);
}

void distributeAfter(PatternRewriter &rewriter, Operation *op_begin,
                     Operation *op_end, tpu::DistributionPattern pattern) {
  // 1. Create pattern params
  auto ctx = rewriter.getContext();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::DistributionPatternAttr::get(ctx, pattern)));

  // 2. Insert DistributionBeginOp
  auto input = op_begin->getResult(0);
  std::vector<Type> types = {input.getType()};
  rewriter.setInsertionPointAfter(op_begin);
  auto begin = rewriter.create<tpu::DistributionBeginOp>(
      module::getLocLike(input, "distribute_begin"), types, ValueRange{input},
      attrs);
  input.replaceAllUsesExcept(begin.getOutput(), begin);

  // 3. Insert DistributionEndOp
  Value output;
  for (auto o : op_end->getResults()) {
    if (o.hasOneUse()) {
      output = o;
      break;
    }
  }
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DistributionEndOp>(
      module::getLocLike(output, "distribute_end"), output.getType(),
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
      splitByDevices<MatMulSliceMerge>(rewriter, op, num_devices);
      break;
    case tpu::DistributionPattern::MatMulSliceMerge2:
      splitByDevices<MatMulSliceMerge2>(rewriter, op, num_devices);
      break;
    case tpu::DistributionPattern::MatMulTopK:
      splitByDevices<MatMulTopK>(rewriter, op, num_devices);
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
    if (module::getNumSubModule() > 0) {
      return;
    }
    if (!(module::isBM1684XFamily() || module::isSG2260Family())) {
      num_device = 1;
    }
    module::setDeviceNum(num_device);
    auto mOp = getOperation();
    auto mainFunc = module::getMainFuncOp(mOp);
    if (num_device > 1) {
      applyPatternOnce<MatMulSliceMerge>(mOp);
      applyPatternOnce<MatMulSliceMerge2>(mOp);
      applyPatternOnce<MatMulTopK>(mOp);
      applyPatternOnce<DoDistributePattern>(mOp);
      if (mainFunc.getOps<tpu::DistributionBeginOp>().empty()) {
        // no pattern find
        num_device = 1;
        module::setDeviceNum(num_device);
      } else {
        applyPatternOnce<DoDistributePattern>(mOp);
      }
    }
    distributeModules(mOp, num_device);
  }

private:
  ModuleOp mOp;
};

std::unique_ptr<OperationPass<ModuleOp>> createDistributePass() {
  return std::make_unique<DistributePass>();
}
} // namespace tpu
} // namespace tpu_mlir
