//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
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
static bool valid_shape(int axis_dims, cvk_fmt_t fmt) {
  int h, w;
  CV18xx::size_to_hw(axis_dims, h, w);
  auto in_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, h, w);
  auto out_shape = CV18xx::tl_shape_t4(1, (int)CV18xx::NPU_NUM, 1, 1);
  auto input_size = CV18xx::lmem_tensor_to_size(in_shape, fmt, 1);
  auto output_size = CV18xx::lmem_tensor_to_size(out_shape, fmt, 1);
  return (uint64_t)(input_size) < ((CV18xx::LMEM_BYTES - 2 * output_size) / 2);
}

class SplitReducePattern : public OpRewritePattern<tpu::ReduceOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    // for ppyoloe reduce 1x48x160x160 --> 1x48x1x1
    if (reduceOp.getMode() == "ReduceL2") {
      return failure();
    }
    std::vector<int64_t> axes_v;
    std::vector<std::vector<int64_t>> outputs_shape_v, new_axes_v;
    auto axes = module::getI64Array(reduceOp.getAxes());
    axes_v.assign(axes->begin(), axes->end());
    int32_t start_axis = axes_v.at(0);
    int32_t end_axis = axes_v.back() + 1;
    auto input_shape = module::getShape(reduceOp.getInput()).vec();
    auto num_axes = axes_v.size();

    int32_t inner_dims =
        std::accumulate(input_shape.begin() + end_axis, input_shape.end(), 1,
                        std::multiplies<int64_t>());
    if (num_axes == 1 || inner_dims > 1) {
      // TODO
      return failure();
    }
    int32_t axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                        input_shape.begin() + end_axis, 1,
                                        std::multiplies<int64_t>());
    auto fmt = CV18xx::getDataType(reduceOp.getInput());
    if (valid_shape(axis_dims, fmt)) {
      return failure();
    }

    for (int32_t i = num_axes - 1; i > 0; i--) {
      int32_t axis = axes_v.at(i);
      axes_v.pop_back();
      new_axes_v.push_back({axis});
      input_shape[axis] = 1;
      outputs_shape_v.push_back(input_shape);
      end_axis = axes_v.back() + 1;
      axis_dims = std::accumulate(input_shape.begin() + start_axis,
                                  input_shape.begin() + end_axis, 1,
                                  std::multiplies<int64_t>());
      if (valid_shape(axis_dims, fmt)) {
        new_axes_v.push_back(axes_v);
        outputs_shape_v.push_back(module::getShape(reduceOp.getOutput()).vec());
        break;
      }
    }

    if (!valid_shape(axis_dims, fmt)) {
      // TODO. Reshape the reduce op to valid
      llvm_unreachable("reduce's axis_dims is too large.");
    }

    // creat Op
    rewriter.setInsertionPointAfter(reduceOp);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    auto eltType = module::getElementType(reduceOp.getOutput());
    auto noneOp = module::getNoneOp(reduceOp);
    operands.push_back(reduceOp.getInput());
    operands.push_back(noneOp);
    operands.push_back(noneOp);
    Value newValue = reduceOp.getOutput();
    for (uint32_t i = 0; i < new_axes_v.size(); i++) {
      auto newType = RankedTensorType::get(outputs_shape_v[i], eltType);
      Location loc = reduceOp.getLoc();
      if (i != new_axes_v.size() - 1) {
        loc = module::getLocLike(reduceOp.getOutput(), std::to_string(i));
      }
      auto newOp = rewriter.create<tpu::ReduceOp>(loc, newType, operands,
                                                  reduceOp->getAttrs());
      newOp->setAttr("axes", rewriter.getI64ArrayAttr(new_axes_v[i]));
      newValue = newOp.getOutput();
      operands[0] = newValue;
    }
    rewriter.replaceOp(reduceOp, {newValue});
    return success();
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
                SplitReluLimitPattern, SplitReducePattern,
                PermuteReorderPattern, PermutePadSwap>(ctx);
};
} // namespace tpu

} // namespace tpu_mlir
