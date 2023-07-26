//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Traits/Traits.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <llvm/Support/Debug.h>

using namespace llvm;

namespace tpu_mlir {
namespace tpu {
// only >= 4MB distribute to multi devices
static const int64_t WEIGHT_LIMIT = 0x400000;

static void distribute(PatternRewriter &rewriter, Operation *op_begin,
                       Operation *op_end, tpu::DistributionPattern pattern) {
  auto name = module::getName(op_begin).str();
  auto ctx = rewriter.getContext();
  auto input = op_begin->getOperand(0);
  std::vector<Type> types = {input.getType()};
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "pattern", tpu::DistributionPatternAttr::get(ctx, pattern)));
  auto begin_loc = NameLoc::get(rewriter.getStringAttr(name + "_begin"));
  rewriter.setInsertionPoint(op_begin);
  auto begin = rewriter.create<tpu::DistributionBeginOp>(
      begin_loc, types, ValueRange{input}, attrs);
  op_begin->setOperand(0, begin.getOutput());

  auto output = op_end->getResult(0);
  auto end_loc = NameLoc::get(rewriter.getStringAttr(name + "_end"));
  rewriter.setInsertionPointAfter(op_end);
  auto end = rewriter.create<tpu::DistributionEndOp>(end_loc, output.getType(),
                                                     ValueRange{output});
  output.replaceAllUsesExcept(end.getOutput(), end);
}

// ======================================
// pattern MatMulSliceMerge
// e.g. ChatGlm2
// ======================================

// e.g. [12, 16, 18] => [12, 16, 9]
static bool isHalfSlice(tpu::SliceOp op) {
  auto offset = module::getI64Array(op.getOffset());
  auto steps = module::getI64Array(op.getSteps());
  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  for (int i = 0; i < in_shape.size(); i++) {
    if (steps->at(i) != 1) {
      return false;
    }
    auto o = offset->at(i);
    if (i == in_shape.size() - 1) {
      if ((o != 0 && o != in_shape[i] / 2) || out_shape[i] != in_shape[i] / 2) {
        return false;
      }
    } else if (o != 0 || in_shape[i] != out_shape[i]) {
      return false;
    }
  }
  return true;
}

static bool isResOp(Operation *op) { return isa<tpu::AddOp, tpu::MulOp>(op); }

static bool isLargeMatMul(Operation *op) {
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

static Operation *cloneOp(PatternRewriter &rewriter, Operation *op,
                          llvm::ArrayRef<int64_t> new_shape,
                          llvm::StringRef suffix) {
  rewriter.setInsertionPointAfter(op);
  auto new_op = rewriter.clone(*op);
  module::setShape(new_op->getResult(0), new_shape);
  module::setLocSuffix(new_op, suffix);
  return new_op;
}

class MatMulSliceMergePattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    if (!isLargeMatMul(op) || module::isOpInDistribution(op)) {
      return failure();
    }
    std::vector<Operation *> users(op->user_begin(), op->user_end());
    if (users.size() != 2) {
      return failure();
    }
    Operation *res_op = nullptr;
    for (auto user : users) {
      auto slice = dyn_cast<tpu::SliceOp>(user);
      if (!slice || !slice->hasOneUse() || !isHalfSlice(slice)) {
        return failure();
      }
      auto next = *slice->user_begin();
      while (next != nullptr) {
        if (isResOp(next)) {
          if (res_op == nullptr) {
            res_op = next;
            continue;
          } else if (next != res_op) {
            return failure();
          }
          break;
        } else if (false == next->hasOneUse() ||
                   !next->hasTrait<trait::SupportElementwise>()) {
          return failure();
        }
        next = *next->user_begin();
      }
    }
    if (!res_op->hasOneUse()) {
      return failure();
    }
    auto next = *res_op->user_begin();
    while (next != nullptr) {
      if (isLargeMatMul(next)) {
        break;
      }
      if (false == next->hasOneUse() ||
          !next->hasTrait<trait::SupportElementwise>()) {
        return failure();
      }
    }
    // Bingo !!
    distribute(rewriter, op, next, tpu::DistributionPattern::MatMulSliceMerge);
    return success();
  }
};

class MatMulTopKPattern : public OpRewritePattern<tpu::MatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    if (!isLargeMatMul(op) || module::isOpInDistribution(op) ||
        !op->hasOneUse()) {
      return failure();
    }
    auto next_op = *op->user_begin();
    auto topk = dyn_cast<tpu::TopKOp>(next_op);
    if (!topk || topk.getK() != 1) {
      return failure();
    }
    // Bingo !!
    distribute(rewriter, op, next_op, tpu::DistributionPattern::MatMulTopK);
    return success();
  }
};

#if 0
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
#endif

// ===================================
// distribute all ops to multi device
// ===================================
class DistributePattern : public OpRewritePattern<tpu::DistributionBeginOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::DistributionBeginOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDone()) {
      return failure();
    }
    auto num_devices = module::getDeviceNum();
    if (op.getPattern() == tpu::DistributionPattern::MatMulSliceMerge) {
      auto next_op = *op->user_begin();
      auto mm0 = cast<tpu::MatMulOp>(next_op);
      auto filterOp = mm0.getRight().getDefiningOp<top::WeightOp>();
      auto filterShape = module::getShape(filterOp.getOutput());
      auto outputShape = module::getShape(mm0.getOutput());
      auto attrs = op->getAttrs();
      auto has_bias = !module::isNone(mm0.getBias());
      auto num_dims = filterShape.size();
      auto N = filterShape[num_dims - 1];
      auto N_half = N / 2;
      auto slice_n = ceiling_func(N_half, num_devices);
      std::vector<Operation *> slices(mm0->user_begin(), mm0->user_end());
      auto slice0Op = cast<tpu::SliceOp>(slices[0]);
      auto offset = module::getI64Array(slice0Op.getOffset());
      if (offset->back() != 0) {
        std::swap(slices[0], slices[1]);
      }
      std::vector<Value> end_operands;
      Operation *end_op = nullptr;
      for (int i = 0; i < num_devices; i++) {
        std::vector<Value> res_operands;
        auto offset = i * slice_n;
        auto length = std::min(slice_n, N_half - offset);
        auto suffix = std::to_string(i);
        // slice one half
        for (int half = 0; half < 2; half++) {
          auto offset_half = offset + half * N_half;
          auto suffix_half = suffix + "_" + std::to_string(half);
          auto newFilter0 = module::opSliceAxis(mm0.getRight(), num_dims - 1,
                                                offset_half, length);
          std::vector<Value> operands;
          operands.push_back(mm0.getInput());
          operands.push_back(newFilter0);
          if (has_bias) {
            auto new_bias = module::opSliceAxis(mm0.getBias(), num_dims - 1,
                                                offset_half, length);
            operands.push_back(new_bias);
          } else {
            operands.push_back(mm0.getBias());
          }
          auto new_loc = module::getLocLike(mm0, suffix_half);
          std::vector<int64_t> new_shape = outputShape;
          new_shape[new_shape.size() - 1] = length;
          auto new_type = module::getTypeLike(mm0.getOutput(), new_shape);
          rewriter.setInsertionPointAfter(mm0);
          auto new_mm0 = rewriter.create<tpu::MatMulOp>(new_loc, new_type,
                                                        operands, attrs);
          Value cur_output = new_mm0.getOutput();
          next_op = *slices[half]->user_begin();
          while (!isResOp(next_op)) {
            auto new_op = cloneOp(rewriter, next_op, new_shape, suffix_half);
            new_op->setOperand(0, cur_output);
            cur_output = new_op->getResult(0);
            next_op = *next_op->user_begin();
          }
          res_operands.push_back(cur_output);
        }
        // res_op: add/mul
        auto new_shape = module::getShape(res_operands[0]);
        auto new_op = cloneOp(rewriter, next_op, new_shape, suffix);
        new_op->setOperands(res_operands);
        Value cur_output = new_op->getResult(0);
        next_op = *next_op->user_begin();
        // matmul op
        while (!isa<tpu::MatMulOp>(next_op)) {
          new_op = cloneOp(rewriter, next_op, new_shape, suffix);
          new_op->setOperand(0, cur_output);
          cur_output = new_op->getResult(0);
          next_op = *next_op->user_begin();
        }
        auto mm1 = cast<tpu::MatMulOp>(next_op);
        auto new_loc = module::getLocLike(next_op, suffix);
        std::vector<Value> operands;
        operands.push_back(cur_output);
        auto newFilter1 =
            module::opSliceAxis(mm1.getRight(), num_dims - 2, offset, length);
        operands.push_back(newFilter1);
        if (module::isNone(mm1.getBias())) {
          operands.push_back(mm1.getBias());
        } else {
          auto bias = mm1.getBias().getDefiningOp<top::WeightOp>();
          operands.push_back(bias.clone(suffix));
        }
        rewriter.setInsertionPointAfter(next_op);
        auto new_mm1 = rewriter.create<tpu::MatMulOp>(
            new_loc, mm1.getOutput().getType(), operands, mm1->getAttrs());
        end_operands.push_back(new_mm1.getOutput());
        if (i == 0) {
          end_op = *next_op->user_begin();
        } else {
          assert(end_op == *next_op->user_begin());
        }
      }
      assert(isa<tpu::DistributionEndOp>(end_op));
      end_op->setOperands(end_operands);
      op.setDone(true);
      return success();
    }
    return failure();
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
    mOp = getOperation();
    ctx = &getContext();
    applyPattern<MatMulSliceMergePattern>();
    applyPattern<MatMulTopKPattern>();
    applyPattern<DistributePattern>();
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
  template <typename T> void applyPattern() {
    RewritePatternSet patterns(ctx);
    patterns.add<T>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDistributePass() {
  return std::make_unique<DistributePass>();
}
} // namespace tpu
} // namespace tpu_mlir
