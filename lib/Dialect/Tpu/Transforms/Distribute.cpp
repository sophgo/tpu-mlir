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
  auto a16_mm = dyn_cast<tpu::A16MatMulOp>(op);

  if (!mm && !a16_mm) {
    return false;
  }

  auto operand = mm ? mm.getOperand(1) : a16_mm.getOperand(1);
  auto is_4bits = a16_mm && a16_mm.getWeightBits() == 4;

  // int4 strategy saves 2 int4 data in one int8
  if (!module::isWeight(operand) ||
      (is_4bits ? 2 : 1) * module::getNumElements(operand) <= WEIGHT_LIMIT) {
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
    auto next_op = *op->user_begin();
    switch (op.getPattern()) {
    case tpu::DistributionPattern::MatMulSliceMerge:
      sliceMergeSplit(dyn_cast<tpu::MatMulOp>(next_op), rewriter, op, num_devices);
      sliceMergeSplit(dyn_cast<tpu::A16MatMulOp>(next_op), rewriter, op, num_devices);
      break;
    case tpu::DistributionPattern::MatMulSliceMerge2:
      sliceMerge2Split(dyn_cast<tpu::MatMulOp>(next_op), rewriter, op, num_devices);
      sliceMerge2Split(dyn_cast<tpu::A16MatMulOp>(next_op), rewriter, op, num_devices);
      break;
    case tpu::DistributionPattern::MatMulTopK:
      topKSplit(dyn_cast<tpu::MatMulOp>(next_op), rewriter, op, num_devices);
      topKSplit(dyn_cast<tpu::A16MatMulOp>(next_op), rewriter, op, num_devices);
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
      applyPatternOnce<MatMulSliceMerge<tpu::MatMulOp>>(mOp);
      applyPatternOnce<MatMulSliceMerge<tpu::A16MatMulOp>>(mOp);
      applyPatternOnce<MatMulSliceMerge2<tpu::MatMulOp>>(mOp);
      applyPatternOnce<MatMulSliceMerge2<tpu::A16MatMulOp>>(mOp);
      applyPatternOnce<MatMulTopK<tpu::MatMulOp>>(mOp);
      applyPatternOnce<MatMulTopK<tpu::A16MatMulOp>>(mOp);
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
