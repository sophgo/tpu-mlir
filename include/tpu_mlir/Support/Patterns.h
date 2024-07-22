//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/Patterns.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

// Common Patterns
namespace tpu_mlir {
namespace patterns {

struct FuseSameOp : public OpRewriterPatternEx3 {
  FuseSameOp(MLIRContext *context)
      : OpRewriterPatternEx3(context,"FuseSameOp",1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                PatternRewriter &rewriter) const override ;
};

// if op =  op + op, fuse to one op. such as top::Reshape
template <typename OpTy>
struct FuseRepeatPattern : public OpRewriterPatternEx<OpTy> {
public:
  FuseRepeatPattern(MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<OpTy>(context, "FuseRepeatPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter) const {
    auto in_op = op.getInput().getDefiningOp();
    if (nullptr == in_op || in_op->hasOneUse() == false) {
      return failure();
    }
    if (!isa<OpTy>(in_op)) {
      return failure();
    }
    op->setOperand(0, in_op->getOperand(0));
    rewriter.eraseOp(in_op);
    return success();
  }
};


// convert op a to op b, not care attributes. such as top::Sequence to top::Reshape
template <typename SourceOp, typename TargetOp>
struct GeneralPattern : public OpRewriterPatternEx<SourceOp> {
  GeneralPattern(MLIRContext *context, const std::string &patternName, int benefit = 1)
      : OpRewriterPatternEx<SourceOp>(context, patternName, benefit) {}

  LogicalResult matchAndRewriteImpl(SourceOp op, PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getOutput().getType(),
                                          op->getOperands(),
                                          std::vector<NamedAttribute>());
    return success();
  }
};

struct SqueezeToReshapePattern : public GeneralPattern<top::SqueezeOp, top::ReshapeOp> {
  SqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<top::SqueezeOp, top::ReshapeOp>(context, "SqueezeToReshapePattern", benefit) {}
};

struct UnsqueezeToReshapePattern : public GeneralPattern<top::UnsqueezeOp, top::ReshapeOp> {
  UnsqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<top::UnsqueezeOp, top::ReshapeOp>(context, "UnsqueezeToReshapePattern", benefit) {}
};

struct TPUSqueezeToReshapePattern : public GeneralPattern<tpu::SqueezeOp, tpu::ReshapeOp> {
  TPUSqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<tpu::SqueezeOp, tpu::ReshapeOp>(context, "TPUSqueezeToReshapePattern", benefit) {}
};

struct TPUUnsqueezeToReshapePattern : public GeneralPattern<tpu::UnsqueezeOp, tpu::ReshapeOp> {
  TPUUnsqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<tpu::UnsqueezeOp, tpu::ReshapeOp>(context, "TPUUnsqueezeToReshapePattern", benefit) {}
};

} // namespace patterns
} // namespace tpu_mlir
