//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "optimize_cv18xx"

using namespace llvm;
using namespace tpu_mlir::backend;
namespace tpu_mlir {

namespace cv18xx {

class MoveConvStrideToEltwiseOpPattern : public RewritePattern {
public:
  MoveConvStrideToEltwiseOpPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasTrait<trait::SupportEarlyStride>()) {
      return failure();
    }
    Operation *nextOp = nullptr;
    int strideH = 0;
    int strideW = 0;
    for (auto &use : op->getResult(0).getUses()) {
      nextOp = use.getOwner();
      if (auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp)) {
        auto attrs = convOp.parseParam();
        int kh = attrs.kh;
        int kw = attrs.kw;
        int sh = attrs.sh;
        int sw = attrs.sw;
        if (kw == 0) {
          return failure();
        }
        if (strideH == 0 || strideW == 0) {
          strideH = sh;
          strideW = sw;
        }
        if (strideH != sh || strideW != sw) {
          LLVM_DEBUG(llvm::errs()
                     << "stride of all successor conv2d should be same\n");
          return failure();
        }
        if (sh == 1 || sw == 1) {
          return failure();
        }
        if (kh != 1 || kw != 1) {
          return failure();
        }
      } else {
        // if one of uses is not 1x1 conv,
        // we cannot do early stride.
        return failure();
      }
    }

    auto shape = module::getShape(op->getResult(0));
    if (shape[2] % strideH != 0 || shape[3] % strideW != 0) {
      // padding case, stop
      return failure();
    }

    for (auto &use : op->getResult(0).getUses()) { // Refactor convOp
      nextOp = use.getOwner();
      auto convOp = dyn_cast<tpu::Conv2DOp>(nextOp);
      convOp->setAttr("strides",
                      rewriter.getI64ArrayAttr({1, 1})); // rewrite strideH
    }

    int on = shape[0];
    int oc = shape[1];
    int oh = shape[2] / strideH;
    int ow = shape[3] / strideW;
    op->setAttr("do_early_stride", rewriter.getBoolAttr(true));
    op->setAttr("early_stride_h", rewriter.getI32IntegerAttr(strideH));
    op->setAttr("early_stride_w", rewriter.getI32IntegerAttr(strideW));
    module::setShape(op->getResult(0), {on, oc, oh, ow});
    // Rename this op to pass the similarity comparison, because its output
    // shape changed
    module::setLocSuffix(op, "early_stride");
    return success();
  }
};

class FuseLeakReluPattern : public OpRewritePattern<tpu::LeakyReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::LeakyReluOp leakyReluOp,
                                PatternRewriter &rewriter) const override {
    assert(leakyReluOp);
    auto preOp = leakyReluOp.getInput().getDefiningOp();
    if (auto convOp = dyn_cast<tpu::Conv2DOp>(preOp)) {
      if (!module::isUniformQuantized(convOp.getOutput()) ||
          !module::isUniformQuantized(leakyReluOp.getOutput())) {
        return failure();
      }
      convOp->setAttr("do_leaky_relu", rewriter.getBoolAttr(true));
      convOp->setAttr("neg_slope", leakyReluOp.getAlphaAttr());
      if (leakyReluOp.getRshift().has_value())
        convOp->setAttr("rshift_pos", leakyReluOp.getRshiftAttr());
      if (leakyReluOp.getMultiplier().has_value())
        convOp->setAttr("multiplier_pos", leakyReluOp.getMultiplierAttr());
      if (leakyReluOp.getRshiftNeg().has_value())
        convOp->setAttr("rshift_neg", leakyReluOp.getRshiftNegAttr());
      if (leakyReluOp.getMultiplierNeg().has_value())
        convOp->setAttr("multiplier_neg", leakyReluOp.getMultiplierNegAttr());
      convOp->setLoc(leakyReluOp.getLoc());
      // remove the relu Op
      rewriter.replaceOp(leakyReluOp, {leakyReluOp.getInput()});
      return success();
    }

    return failure();
  }
};

class SplitReluLimitPattern : public RewritePattern {
public:
  SplitReluLimitPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);
    if (isa<ReturnOp>(op)) {
      return failure();
    }
    if (op->hasTrait<trait::SupportFuseRelu>() &&
        module::getStorageType(op->getResult(0)).isBF16()) {
      auto max = op->getAttr("relu_limit").cast<FloatAttr>().getValueAsDouble();
      if (max == -1 || !op->getAttr("do_relu")) {
        return failure();
      }
      op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.));
      auto uses = op->getResult(0).getUses();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("min", rewriter.getF64FloatAttr(0.)));
      attrs.push_back(
          rewriter.getNamedAttr("max", rewriter.getF64FloatAttr(max)));
      auto tensor_type = op->getResult(0).getType().cast<RankedTensorType>();
      auto newType =
          RankedTensorType::get(tensor_type.getShape(), rewriter.getBF16Type());
      auto newOp = rewriter.create<tpu::ClipOp>(op->getLoc(), newType,
                                                op->getResults(), attrs);
      module::setLocSuffix(op, "0");
      for (auto &use : uses) {
        auto useOp = use.getOwner();
        int32_t num = useOp->getNumOperands();
        for (int32_t i = 0; i < num; i++) {
          if (useOp->getOperand(i) == op->getResult(0)) {
            useOp->setOperand(i, newOp.getOutput());
          }
        }
      }
      return success();
    } else {
      return failure();
    }
  }
};

} // namespace cv18xx

namespace tpu {
using namespace cv18xx;
void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns) {
  auto ctx = patterns->getContext();
  patterns->add<FuseLeakReluPattern, MoveConvStrideToEltwiseOpPattern,
                SplitReluLimitPattern, PermuteReorderPattern, PermutePadSwap>(
      ctx);
};
} // namespace tpu

} // namespace tpu_mlir
