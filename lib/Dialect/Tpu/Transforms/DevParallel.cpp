//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/Distribute.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;
namespace tpu_mlir {
namespace tpu {
// only >= 4MB distribute to multi devices
static const int64_t WEIGHT_LIMIT = 0x400000;

// num_head is used for splitting matmul that dim is not divisible by num_device
void distribute(PatternRewriter &rewriter, std::vector<Operation *> ops_begin,
                std::vector<Operation *> ops_end, tpu::DevPattern pattern,
                std::vector<int64_t> &begin_methods,
                std::vector<int64_t> &end_methods, int num_head) {
  // 1. Create pattern params
  auto ctx = rewriter.getContext();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("pattern", tpu::DevPatternAttr::get(ctx, pattern)));
  attrs.push_back(rewriter.getNamedAttr(
      "begin_methods",
      rewriter.getI64ArrayAttr(llvm::ArrayRef(begin_methods))));
  attrs.push_back(
      rewriter.getNamedAttr("num_head", rewriter.getI64IntegerAttr(num_head)));

  // 2. Insert DevBeginOp
  std::vector<Value> inputs;
  std::vector<Type> types;
  std::vector<Location> locs;
  Value opd = ops_begin[0]->getOperand(0);
  if (isa<tpu::MatMulOp, tpu::A16MatMulOp>(ops_begin[0])) {
    int num_opds = ops_begin[0]->getNumOperands();
    opd = ops_begin[0]->getOperand(num_opds - 1);
  } else if (isa<tpu::AddOp>(ops_begin[0])) {
    auto op0 = ops_begin[0]->getOperand(0).getDefiningOp();
    auto op1 = ops_begin[0]->getOperand(1).getDefiningOp();
    if (op1->isBeforeInBlock(op0)) {
      opd = ops_begin[0]->getOperand(1);
    }
  }
  for (auto op : ops_begin) {
    auto input = op->getOperand(0);
    if (isa<tpu::GatherOp>(op)) {
      input = op->getOperand(1);
    } else if (isa<tpu::MatMulOp>(op)) {
      if (!isa<top::NoneOp>(op->getOperand(2).getDefiningOp())) {
        input = op->getOperand(2);
      }
    } else if (isa<tpu::AddOp>(op)) {
      for (auto in : op->getOperands()) {
        if (!isa<tpu::MulConstOp, tpu::MatMulOp, tpu::A16MatMulOp>(
                in.getDefiningOp())) {
          input = in;
        }
      }
    } else if (isa<tpu::FAttentionOp>(op)) {
      for (auto in : op->getOperands()) {
        if (isa<top::InputOp>(in.getDefiningOp())) {
          input = in;
        }
      }
    }
    if (opd.getDefiningOp()->isBeforeInBlock(input.getDefiningOp())) {
      opd = input;
    }
    auto type = input.getType();
    auto loc = module::getLocLike(op, "distribute_begin");
    inputs.push_back(input);
    types.push_back(type);
    locs.push_back(loc);
  }
  for (auto op : ops_begin) {
    if (op->isBeforeInBlock(opd.getDefiningOp())) {
      op->moveAfter(opd.getDefiningOp());
    }
  }
  auto begin_loc = FusedLoc::get(ctx, locs);
  rewriter.setInsertionPointAfterValue(opd);
  auto begin =
      rewriter.create<tpu::DevBeginOp>(begin_loc, types, inputs, attrs);

  for (size_t i = 0; i < ops_begin.size(); ++i) {
    int index = 0;
    for (auto [idx, in] : llvm::enumerate(ops_begin[i]->getOperands())) {
      if (in == begin.getInputs()[i]) {
        index = idx;
        break;
      }
    }
    ops_begin[i]->setOperand(index, begin.getOutputs()[i]);
  }

  // 3. Insert DevEndOp
  attrs.clear();
  inputs.clear();
  types.clear();
  locs.clear();

  attrs.push_back(
      rewriter.getNamedAttr("pattern", tpu::DevPatternAttr::get(ctx, pattern)));
  attrs.push_back(rewriter.getNamedAttr(
      "end_methods", rewriter.getI64ArrayAttr(llvm::ArrayRef(end_methods))));

  for (auto op : ops_end) {
    for (auto o : op->getResults()) {
      inputs.push_back(o);
      types.push_back(o.getType());
      auto loc = module::getLocLike(op, "distribute_end");
      locs.push_back(loc);
      break;
    }
  }
  auto end_loc = FusedLoc::get(ctx, locs);
  rewriter.setInsertionPointAfter(ops_end[0]);
  auto end = rewriter.create<tpu::DevEndOp>(end_loc, types, inputs, attrs);

  for (size_t i = 0; i < ops_end.size(); ++i) {
    inputs[i].replaceUsesWithIf(end.getOutputs()[i], [&](OpOperand &use) {
      return (use.getOwner() != end &&
              !isa<tpu::ConcatOp, tpu::MatMulOp, tpu::PermuteOp,
                   tpu::UnsqueezeOp, tpu::FAttentionOp>(use.getOwner()));
    });
  }
}

// insert DevBeginOp before op_begin
// insert DevEndOp after op_end
void distribute(PatternRewriter &rewriter, Operation *op_begin,
                Operation *op_end, tpu::DevPattern pattern) {
  auto ctx = rewriter.getContext();
  auto input = op_begin->getOperand(0);
  std::vector<Type> types = {input.getType()};
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("pattern", tpu::DevPatternAttr::get(ctx, pattern)));
  std::vector<int64_t> begin_methods{1};
  attrs.push_back(rewriter.getNamedAttr(
      "begin_methods",
      rewriter.getI64ArrayAttr(llvm::ArrayRef(begin_methods))));
  rewriter.setInsertionPoint(op_begin);
  auto begin = rewriter.create<tpu::DevBeginOp>(
      module::getLocLike(op_begin, "distribute_begin"), types,
      ValueRange{input}, attrs);
  op_begin->setOperand(0, begin.getOutputs()[0]);

  Value output;
  for (auto o : op_end->getResults()) {
    if (o.hasOneUse()) {
      output = o;
      break;
    }
  }

  attrs.clear();
  std::vector<int64_t> end_methods{2};
  attrs.push_back(
      rewriter.getNamedAttr("pattern", tpu::DevPatternAttr::get(ctx, pattern)));
  attrs.push_back(rewriter.getNamedAttr(
      "end_methods", rewriter.getI64ArrayAttr(llvm::ArrayRef(end_methods))));
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DevEndOp>(
      module::getLocLike(output, "distribute_end"), output.getType(),
      ValueRange{output}, attrs);
  output.replaceAllUsesExcept(end.getOutputs()[0], end);
}

// insert DevBeginOp after op_begin
// insert DevEndOp after op_end
void distributeAfter(PatternRewriter &rewriter, Operation *op_begin,
                     Operation *op_end, tpu::DevPattern pattern) {
  // 1. Create pattern params
  auto ctx = rewriter.getContext();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("pattern", tpu::DevPatternAttr::get(ctx, pattern)));

  // 2. Insert DevBeginOp
  auto input = op_begin->getResult(0);
  std::vector<Type> types = {input.getType()};
  rewriter.setInsertionPointAfter(op_begin);
  auto begin = rewriter.create<tpu::DevBeginOp>(
      module::getLocLike(input, "distribute_begin"), types, ValueRange{input},
      attrs);
  input.replaceAllUsesExcept(begin.getOutputs()[0], begin);

  // 3. Insert DevEndOp
  Value output;
  for (auto o : op_end->getResults()) {
    if (o.hasOneUse()) {
      output = o;
      break;
    }
  }
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DevEndOp>(
      module::getLocLike(output, "distribute_end"), output.getType(),
      ValueRange{output}, attrs);
  output.replaceAllUsesExcept(end.getOutputs()[0], end);
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
class DoDistributePattern : public OpRewriterPatternEx<tpu::DevBeginOp> {
public:
  DoDistributePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::DevBeginOp>(context) {
    num_devices = module::getDeviceNum();
  }

  LogicalResult matchAndRewriteImpl(tpu::DevBeginOp op,
                                    PatternRewriter &rewriter) const override {

    if (op.getDone()) {
      return failure();
    }
    auto next_op = *op->user_begin();
    auto mode = module::getMode();
    if (mode == module::Mode::F16 || mode == module::Mode::BF16) {
      switch (op.getPattern()) {
      case tpu::DevPattern::MatMulSliceMerge:
        sliceMergeSplit(dyn_cast<tpu::MatMulOp>(next_op), rewriter, op,
                        num_devices);
        break;
      case tpu::DevPattern::AttentionSliceMerge:
        sliceAttentionMergeSplit(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulSliceMerge2:
        sliceMerge2Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::AttentionSliceMerge2:
        sliceAttentionMerge2Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulSliceMerge3:
        sliceMerge3Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::FAttentionSliceMerge:
        sliceFAttentionMergeSplit(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulTopK:
        topKSplit(dyn_cast<tpu::MatMulOp>(next_op), rewriter, op, num_devices);
        break;
      case tpu::DevPattern::EmbeddingSliceMerge:
        embeddingMergeSplit(rewriter, op, num_devices);
        break;
      default:
        return failure();
      }
    } else if (mode == module::Mode::W8F16 || mode == module::Mode::W8BF16 ||
               mode == module::Mode::W4F16 || mode == module::Mode::W4BF16) {
      switch (op.getPattern()) {
      case tpu::DevPattern::MatMulSliceMerge:
        sliceMergeSplit(dyn_cast<tpu::A16MatMulOp>(next_op), rewriter, op,
                        num_devices);
        break;
      case tpu::DevPattern::AttentionSliceMerge:
        sliceAttentionMergeSplit(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulSliceMerge2:
        sliceMerge2Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::AttentionSliceMerge2:
        sliceAttentionMerge2Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulSliceMerge3:
        sliceMerge3Split(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::FAttentionSliceMerge:
        sliceFAttentionMergeSplit(rewriter, op, num_devices);
        break;
      case tpu::DevPattern::MatMulTopK:
        topKSplit(dyn_cast<tpu::A16MatMulOp>(next_op), rewriter, op,
                  num_devices);
        break;
      default:
        return failure();
      }
    } else {
      llvm_unreachable("Not supported quantization mode");
    }
    op.setDone(true);
    return success();
  }

  bool shouldPrint(tpu::DevBeginOp op) const override { return false; }

private:
  int64_t num_devices;
};

class DevParallelPass : public DevParallelBase<DevParallelPass> {
public:
  DevParallelPass() {}
  void runOnOperation() override {
    if (module::getNumSubModule() > 0) {
      return;
    }
    auto num_device = module::getDeviceNum();
    auto mOp = getOperation();
    auto mainFunc = module::getMainFuncOp(mOp);
    auto mode = module::getMode();
    if (num_device > 1) {
      if (mode == module::Mode::F16 || mode == module::Mode::BF16) {
        module::applyPatternOnce<MatMulSliceMerge3>(mOp);
        module::applyPatternOnce<MatMulSliceMerge<tpu::MatMulOp>>(mOp);
        module::applyPatternOnce<MatMulSliceMerge2<tpu::MatMulOp>>(mOp);
        module::applyPatternOnce<MatMulTopK<tpu::MatMulOp>>(mOp);
        module::applyPatternOnce<AttentionSliceMerge<tpu::MatMulOp>>(mOp);
        module::applyPatternOnce<AttentionSliceMerge2<tpu::MatMulOp>>(mOp);
        module::applyPatternOnce<FAttentionSliceMerge>(mOp);
        // module::applyPatternOnce<EmbeddingSliceMerge>(mOp);
      } else if (mode == module::Mode::W8F16 || mode == module::Mode::W8BF16 ||
                 mode == module::Mode::W4F16 || mode == module::Mode::W4BF16) {
        module::applyPatternOnce<MatMulSliceMerge<tpu::A16MatMulOp>>(mOp);
        module::applyPatternOnce<MatMulSliceMerge2<tpu::A16MatMulOp>>(mOp);
        module::applyPatternOnce<MatMulTopK<tpu::A16MatMulOp>>(mOp);
        module::applyPatternOnce<AttentionSliceMerge<tpu::A16MatMulOp>>(mOp);
        module::applyPatternOnce<AttentionSliceMerge2<tpu::A16MatMulOp>>(mOp);
        module::applyPatternOnce<FAttentionSliceMerge>(mOp);
      } else {
        llvm_unreachable("Not supported quantization mode");
      }
      if (mainFunc.getOps<tpu::DevBeginOp>().empty()) {
        // no pattern find, copy the whole modules num_device times
        num_device = 1;
      } else {
        module::applyPatternOnce<DoDistributePattern>(mOp);
      }
    }
    distributeModules(mOp, num_device);
  }

private:
  ModuleOp mOp;
};

void dump(Operation *op) {
  Value res;
  if (op->getNumResults() > 0) {
    res = op->getResult(0);
    op->dump();
  }

  std::vector<Value> opds(op->operand_begin(), op->operand_end());
  printf("### It has %ld operands.\n", opds.size());
  for (auto opd : opds) {
    opd.dump();
  }

  std::vector<Operation *> users(op->user_begin(), op->user_end());
  printf("### It has %ld users.\n", users.size());
  for (auto user : users) {
    user->dump();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createDevParallelPass() {
  return std::make_unique<DevParallelPass>();
}
} // namespace tpu
} // namespace tpu_mlir
