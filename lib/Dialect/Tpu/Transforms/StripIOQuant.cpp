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

namespace tpu_mlir {
namespace tpu {

struct StripInputQuantTpuCastPattern : public OpRewritePattern<tpu::CastOp> {
  StripInputQuantTpuCastPattern(MLIRContext *context)
      : OpRewritePattern<tpu::CastOp>(context) {}
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (auto inputOp = op.getInput().getDefiningOp<top::InputOp>()) {
      auto out = inputOp.getOutput();
      if (!out.hasOneUse()) {
        return failure();
      }
      if (!module::isUniformQuantized(op.getOutput())) {
        return failure();
      }
      out.setType(op.getResult().getType());
      rewriter.replaceOp(op, out);
      return success();
    }
    // for case input -> reshape -> cast -> any op
    if (auto reshapeOp = op.getInput().getDefiningOp<tpu::ReshapeOp>()) {
      if (!reshapeOp.getResult().hasOneUse()) {
        return failure();
      }
      auto inputOp = reshapeOp.getInput().getDefiningOp<top::InputOp>();
      if (!inputOp) {
        return failure();
      }
      auto new_ele_type = module::getElementType(op.getResult());
      auto input_new_type = RankedTensorType::get(
          inputOp.getResult().getType().cast<RankedTensorType>().getShape(),
          new_ele_type);
      inputOp.getResult().setType(input_new_type);
      auto reshape_new_type = RankedTensorType::get(
          reshapeOp.getResult().getType().cast<RankedTensorType>().getShape(),
          new_ele_type);
      reshapeOp.getResult().setType(reshape_new_type);
      rewriter.replaceOp(op, reshapeOp.getResult());
      return success();
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
      inputOp->getResult(0).setType(op.getResults()[0].getType());
      rewriter.replaceOp(op, inputOp.getResult());
      return success();
    }
    // for case input -> reshape -> cast -> any op
    if (auto reshapeOp = op.getInputs()[0].getDefiningOp<tpu::ReshapeOp>()) {
      if (!reshapeOp.getResult().hasOneUse()) {
        return failure();
      }
      auto inputOp = reshapeOp.getInput().getDefiningOp<top::InputOp>();
      if (!inputOp) {
        return failure();
      }
      auto new_ele_type = module::getElementType(op.getResults()[0]);
      auto input_new_type = RankedTensorType::get(
          inputOp.getResult().getType().cast<RankedTensorType>().getShape(),
          new_ele_type);
      inputOp.getResult().setType(input_new_type);
      auto reshape_new_type = RankedTensorType::get(
          reshapeOp.getResult().getType().cast<RankedTensorType>().getShape(),
          new_ele_type);
      reshapeOp.getResult().setType(reshape_new_type);
      rewriter.replaceOp(op, reshapeOp.getResult());
      return success();
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
      auto in = op.getInput();
      if (!module::isUniformQuantized(in)) {
        return failure();
      }
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
