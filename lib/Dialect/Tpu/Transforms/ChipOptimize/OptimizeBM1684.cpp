//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include <cstdint>

using namespace llvm;
namespace tpu_mlir {

namespace bm1684 {
class CastWithoutScalePattern : public OpRewritePattern<tpu::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {
    if (!module::isBM1684Family()) {
      return failure();
    }
    if (!op.getWithScale()) {
      return failure();
    }

    auto input = op.getInput();
    auto output = op.getOutput();
    bool qInput = module::isUniformQuantized(input);
    bool qOutput = module::isUniformQuantized(output);
    if (!qInput && !qOutput) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    auto name = module::getName(op.getOutput());
    if (qInput && !qOutput) {
      auto scale = module::getUniformQuantizedType(input).getScale();
      if (scale == 1.f) {
        return failure();
      }
      auto cast_loc =
          NameLoc::get(rewriter.getStringAttr(name.str() + "_new_cast"));
      auto new_type = output.getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("with_scale", rewriter.getBoolAttr(false)));
      auto cast_op = rewriter.create<tpu::CastOp>(cast_loc, new_type,
                                                  ValueRange{input}, attrs);
      attrs.clear();
      attrs.push_back(
          rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(scale)));
      auto mul_op = rewriter.create<tpu::MulConstOp>(
          op.getLoc(), new_type, ValueRange{cast_op.getOutput()}, attrs);
      op.replaceAllUsesWith(mul_op.getOperation());
      op.erase();
      return success();
    } else if (!qInput && qOutput) {
      auto orin_scale = module::getUniformQuantizedType(output).getScale();
      if (orin_scale == 1.f) {
        return failure();
      }
      auto scale = 1.f / orin_scale;
      auto mul_loc =
          NameLoc::get(rewriter.getStringAttr(name.str() + "_mul_scale"));
      auto new_type = input.getType();
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(scale)));
      auto mul_op = rewriter.create<tpu::MulConstOp>(mul_loc, new_type,
                                                     ValueRange{input}, attrs);
      new_type = output.getType();
      attrs.clear();
      attrs.push_back(
          rewriter.getNamedAttr("with_scale", rewriter.getBoolAttr(false)));
      auto cast_op = rewriter.create<tpu::CastOp>(
          op.getLoc(), new_type, ValueRange{mul_op.getOutput()}, attrs);
      op.replaceAllUsesWith(cast_op.getOperation());
      op.erase();
      return success();
    }
    return failure();
  }
};

} // namespace bm1684

namespace tpu {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    CastWithoutScalePattern
  >(patterns->getContext());
  // clang-format on
};
} // namespace tpu

} // namespace tpu_mlir
