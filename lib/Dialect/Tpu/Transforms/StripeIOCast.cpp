//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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

struct StripInputCastPattern : public OpRewritePattern<tpu::CastOp> {
  StripInputCastPattern(MLIRContext *context)
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

struct StripOutputCastPattern : public OpRewritePattern<tpu::CastOp> {
  StripOutputCastPattern(MLIRContext *context)
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

class StripIOCastPass : public StripIOCastBase<StripIOCastPass> {
public:
  StripIOCastPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    if (func.getName() != "main") {
      return;
    }
    auto ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    if (quant_input)
      patterns.add<StripInputCastPattern>(ctx);
    if (quant_output)
      patterns.add<StripOutputCastPattern>(ctx);
    applyPatternsAndFoldGreedily(func, std::move(patterns));
    Module::updateModuleTypes(Module::getModuleOp(func));
  }
};

std::unique_ptr<OperationPass<FuncOp>> createStripIOCast() {
  return std::make_unique<StripIOCastPass>();
}
} // namespace tpu
} // namespace tpu_mlir
