//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"

using namespace mlir;
namespace tpu_mlir {
namespace tpu {

struct StripInputQuantTpuCastPattern : public OpRewritePattern<tpu::CastOp> {
  StripInputQuantTpuCastPattern(MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context) {}
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (auto inputOp = op.getInput().getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      inputOp.getResult().setType(op.getResult().getType());
      rewriter.replaceOp(op, inputOp.getResult());
      return success();
    }
    // for case input -> reshape -> cast -> any op
    if(auto reshapeOp = op.getInput().getDefiningOp<tpu::ReshapeOp>()) {
      if (!reshapeOp.getResult().hasOneUse()) {
        return failure();
      }
      auto inputOp = reshapeOp.getInput().getDefiningOp<top::InputOp>();
      if (!inputOp) {
        return failure();
      }
      auto new_type = op.getResult().getType();
      inputOp.getResult().setType(new_type);
      reshapeOp.getResult().setType(new_type);
      rewriter.replaceOp(op, reshapeOp.getResult());
    }
    return failure();
  };
};

struct StripInputQuantCpuCastPattern
    : public OpRewritePattern<tpu::GenericCpuOp> {
  StripInputQuantCpuCastPattern(MLIRContext *context)
      : OpRewritePattern<tpu::GenericCpuOp>(context) {}
  LogicalResult matchAndRewrite(tpu::GenericCpuOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getCpuOpName() != "quant") {
      return failure();
    }
    if (auto inputOp = op.getInputs()[0].getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      inputOp.getResult().setType(op.getResults()[0].getType());
      rewriter.replaceOp(op, inputOp.getResult());
      return success();
    }
    // for case input -> reshape -> cast -> any op
    if(auto reshapeOp = op.getInputs()[0].getDefiningOp<tpu::ReshapeOp>()) {
      if (!reshapeOp.getResult().hasOneUse()) {
        return failure();
      }
      auto inputOp = reshapeOp.getInput().getDefiningOp<top::InputOp>();
      if (!inputOp) {
        return failure();
      }
      auto new_type = op.getResults()[0].getType();
      inputOp.getResult().setType(new_type);
      reshapeOp.getResult().setType(new_type);
      rewriter.replaceOp(op, reshapeOp.getResult());
    }
    return failure();
  };
};

struct StripOutputQuantTpuCastPattern : public OpRewritePattern<tpu::CastOp> {
  StripOutputQuantTpuCastPattern(MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context) {}
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getOutput().hasOneUse() &&
        isa<ReturnOp>(op.getOutput().use_begin().getUser())) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }
    return failure();
  };
};

struct StripOutputQuantCpuCastPattern
    : public OpRewritePattern<tpu::GenericCpuOp> {
  StripOutputQuantCpuCastPattern(MLIRContext *context)
      : OpRewritePattern<tpu::GenericCpuOp>(context) {}
  LogicalResult matchAndRewrite(tpu::GenericCpuOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isCV18xx()) {
      if (op.getOutputs()[0].hasOneUse() &&
          isa<ReturnOp>(op.getOutputs()[0].use_begin().getUser())) {
        rewriter.replaceOp(op, op.getInputs()[0]);
        return success();
      }
    }
    return failure();
  };
};

class StripIOQuantPass : public StripIOQuantBase<StripIOQuantPass> {
public:
  StripIOQuantPass() {}
  void runOnOperation() override {
    auto func = module::getMainFuncOp();
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    if (quant_input) {
      patterns.add<StripInputQuantTpuCastPattern>(ctx);
      patterns.add<StripInputQuantCpuCastPattern>(ctx);
    }
    if (quant_output) {
      patterns.add<StripOutputQuantTpuCastPattern>(ctx);
      patterns.add<StripOutputQuantCpuCastPattern>(ctx);
    }
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripIOQuant() {
  return std::make_unique<StripIOQuantPass>();
}
} // namespace tpu
} // namespace tpu_mlir
