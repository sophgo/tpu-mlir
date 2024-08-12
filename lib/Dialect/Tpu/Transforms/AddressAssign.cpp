//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "AddressAssign/BMAddressAssign.h"
#include "AddressAssign/CVAddressAssign.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

extern void populateGlobalBufferBM168xPatterns(RewritePatternSet *patterns);

class ConcatFusePattern : public OpRewritePattern<tpu::ConcatOp> {
public:
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOnlyMerge()) {
      return failure();
    }
    if (op.getDoRelu()) {
      return failure();
    }
    if (module::isBM1684Family() &&
        module::isUniformQuantized(op.getOutput())) {
      // 1684 4N mode not support
      return failure();
    }
    auto shape = module::getShape(op.getOutput());
    int outer_dim = std::accumulate(shape.begin(), shape.begin() + op.getAxis(),
                                    1, std::multiplies<int64_t>());
    if (outer_dim != 1) {
      return failure();
    }
    int multi_use_times = 0;
    for (auto in : op.getInputs()) {
      if (module::isWeight(in)) {
        return failure();
      }
      int same_op_times = 0;
      if (in.hasOneUse() == false) {
        multi_use_times++;
        for (auto v : op.getInputs()) {
          if (in == v) {
            same_op_times++;
          }
        }

        // if value is used by multiple ConcatOp, only one ConcatOp should be
        // setOnlyMerge
        for (auto user : in.getUsers()) {
          if (auto other_cat = dyn_cast<tpu::ConcatOp>(user)) {
            if (other_cat.getOnlyMerge()) {
              return failure();
            }
          }
        }

        if (same_op_times > 1) {
          return failure();
        }
      }
      if (multi_use_times > 2) {
        return failure();
      }
      auto in_op = in.getDefiningOp();
      if (in_op == nullptr) {
        // return failure();
      } else if (isa<tpu::ConcatOp>(in_op)) {
        return failure();
      } else if (auto rshape = dyn_cast<tpu::ReshapeOp>(in_op)) {
        auto in2 = rshape.getInput();
        if (in2.getDefiningOp() == nullptr || in2.hasOneUse() == false) {
          return failure();
        }
      } else if (auto sliceOp = dyn_cast<tpu::SliceOp>(in_op)) {
        auto p = sliceOp.parseParam();
        if (p.fusible) {
          return failure();
        }
      }
    }
    op.setOnlyMerge(true);
    return success();
  }
};

// concat(concat(a,b),c) => concat(a,b,c)
class ConcatMergePattern : public OpRewritePattern<tpu::ConcatOp> {
public:
  using OpRewritePattern<tpu::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684XFamily()) {
      return failure();
    }
    if (module::isOpInBlock(op)) {
      return failure();
    }
    auto relu = op.getDoRelu();
    auto axis = op.getAxis();
    std::vector<Operation *> cat_ops;
    std::vector<Value> operands;
    bool fix = false;
    for (auto in : op.getInputs()) {
      auto concat_in = dyn_cast_or_null<tpu::ConcatOp>(in.getDefiningOp());
      if (concat_in && concat_in->hasOneUse() &&
          concat_in.getDoRelu() == relu && concat_in.getAxis() == axis) {
        for (auto pre_in : concat_in.getInputs()) {
          operands.push_back(pre_in);
        }
        cat_ops.push_back(in.getDefiningOp());
        fix = true;
      } else {
        operands.push_back(in);
      }
    }
    if (fix == false) {
      return failure();
    }
    op->setOperands(operands);
    for (auto cop : cat_ops) {
      rewriter.eraseOp(cop);
    }
    return success();
  }
};

class AddressAssignPass : public AddressAssignBase<AddressAssignPass> {
public:
  AddressAssignPass() {}
  void runOnOperation() override {
    if (!module::isState(module::State::TPU_DIVIDED)) {
      llvm_unreachable("module should be divided");
    }
    module::removeUnusedOp();
    auto modules = module::getAllModules();
    for (auto s : *modules) {
      if (module::isCV18xx()) {
        CVAddressAssign addr_assign;
        addr_assign.assign(s, reuse_addr, merge_weight, compress_weight,
                           weight_map_file);
      } else {
        RewritePatternSet patterns(s.getContext());
        populateGlobalBufferBM168xPatterns(&patterns);
        applyPatternsAndFoldGreedily(s, std::move(patterns));
        module::applyPatternOnce<ConcatMergePattern>(s);
        module::applyPatternOnce<ConcatFusePattern>(s);
        BMAddressAssign addr_assign;
        addr_assign.assign(s, reuse_addr);
      }
    }
    module::setState(module::State::TPU_ADDRESSED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddressAssignPass() {
  return std::make_unique<AddressAssignPass>();
}
} // namespace tpu
} // namespace tpu_mlir
