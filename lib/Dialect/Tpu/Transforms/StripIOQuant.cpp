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
    if (auto inputOp = op.input().getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      inputOp.getResult().setType(op.getResult().getType());
      rewriter.replaceOp(op, inputOp.getResult());
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
    if (op.operation_name() != "quant") {
      return failure();
    }
    if (auto inputOp = op.inputs()[0].getDefiningOp<top::InputOp>()) {
      if (!inputOp.getResult().hasOneUse())
        return failure();
      inputOp.getResult().setType(op.getResult().getType());
      rewriter.replaceOp(op, inputOp.getResult());
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

    if (op.output().hasOneUse() &&
        isa<ReturnOp>(op.output().use_begin().getUser())) {
      rewriter.replaceOp(op, op.input());
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

    if (op.output().hasOneUse() &&
        isa<ReturnOp>(op.output().use_begin().getUser())) {
      rewriter.replaceOp(op, op.inputs()[0]);
      return success();
    }
    return failure();
  };
};

class StripIOQuantPass : public StripIOQuantBase<StripIOQuantPass> {
public:
  StripIOQuantPass() {}
  void runOnOperation() override {
    auto func = Module::getMainFuncOp();
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
    Module::updateModuleTypes();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createStripIOQuant() {
  return std::make_unique<StripIOQuantPass>();
}
} // namespace tpu
} // namespace tpu_mlir
