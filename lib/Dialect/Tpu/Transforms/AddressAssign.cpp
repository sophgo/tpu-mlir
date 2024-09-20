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
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

extern void populateGlobalBufferBM168xPatterns(RewritePatternSet *patterns);


class ConcatFusePattern  : public OpRewriterPatternEx<tpu::ConcatOp> {
  public:
  ConcatFusePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ConcatOp>(context,"ConcatFusePattern") {}

  LogicalResult matchAndRewriteImpl(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.supportInplace()){
      return failure();
    }
    op.setOnlyMerge(true);
    return success();
  }
  bool shouldPrint(tpu::ConcatOp op) const override { return false;}
};


class ConcatMergePattern  : public OpRewriterPatternEx<tpu::ConcatOp> {
  public:
  ConcatMergePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ConcatOp>(context,"ConcatMergePattern") {}

  LogicalResult matchAndRewriteImpl(tpu::ConcatOp op,
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
        if (concat_in.getAxis()!= op.getAxis()) {
          fix = false;
          break;
        }
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
  bool shouldPrint(tpu::ConcatOp op) const override { return false;}
};

class Concat_SlicePattern  : public OpRewriterPatternEx<tpu::ConcatOp> {
  public:
  Concat_SlicePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ConcatOp>(context,"Concat_SlicePattern") {}

  LogicalResult matchAndRewriteImpl(tpu::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getOnlyMerge()) {
      return failure();
    }
    for (auto in : op.getInputs()) {
      auto in_op = in.getDefiningOp();
      auto reshape = dyn_cast_or_null<tpu::ReshapeOp>(in_op);
      if (reshape){
        auto reshapeOp = dyn_cast<tpu::ReshapeOp>(in_op);
        if (auto sliceOp = dyn_cast<tpu::SliceOp>(reshapeOp.getInput().getDefiningOp())) {
          auto p = sliceOp.parseParam();
          if (p.fusible) {
            op.setOnlyMerge(false);
          }
        }
      }
    }
    return success();
  }
  bool shouldPrint(tpu::ConcatOp op) const override { return false;}
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
        module::applyPatternOnce<Concat_SlicePattern>(s);
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
