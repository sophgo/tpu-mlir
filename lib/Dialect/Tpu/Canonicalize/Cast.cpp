//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

namespace tpu_mlir {
namespace tpu {

class SimplifyRedundantCast : public OpRewriterPatternEx<tpu::CastOp> {
public:
  SimplifyRedundantCast(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::CastOp>(context, "SimplifyRedundantCast") {}

  LogicalResult matchAndRewriteImpl(tpu::CastOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto in_type = in.getType();
    auto out_type = op.getOutput().getType();
    if (type_need_cast(in_type, out_type) == false) {
      // for example, int32 cast int32 => remove this one cast
      rewriter.replaceOp(op, {in});
      return success();
    }
    auto castInputOp =
        dyn_cast<tpu::CastOp>(module::getOriValue(in).getDefiningOp());
    if (!castInputOp || castInputOp->hasOneUse() == false) {
      return failure();
    }
    auto pre_cast_in = castInputOp.getInput();
    if (out_type == pre_cast_in.getType()) {
      // for example, int32 cast f16 cast int32 => remove these two cast
      rewriter.replaceOp(op, {pre_cast_in});
      return success();
    }
    bool is_qtype_out = module::isUniformQuantized(out_type);
    bool is_qtype_in = module::isUniformQuantized(in);
    bool is_qtype_pre_in = module::isUniformQuantized(pre_cast_in);
    if (false == is_qtype_out && false == is_qtype_pre_in) {
      // for example, int32 cast int8 cast f16 => int32 cast f16
      op->setOperand(0, pre_cast_in);
      rewriter.eraseOp(castInputOp);
      return success();
    }
    if (is_qtype_out && false == is_qtype_in && false == is_qtype_pre_in) {
      auto pre_stype = module::getStorageType(pre_cast_in);
      if (pre_stype.isa<mlir::FloatType>()) {
        // for example, f32 cast f16, f16 cast int8 => f32 cast int8
        op->setOperand(0, pre_cast_in);
        rewriter.eraseOp(castInputOp);
        return success();
      }
      if (pre_stype.isIntOrIndex()) {
        // for example, int32 cast f32, f32 cast int8 => int32 requant to int8
        auto qtype = module::getUniformQuantizedType(out_type);
        int32_t multiplier;
        int32_t shift;
        std::vector<NamedAttribute> attrs;
        get_scale_and_shift(1.0 / qtype.getScale(), multiplier, shift, 32);
        auto ctx = op.getContext();
        attrs.push_back(rewriter.getNamedAttr(
            "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
        attrs.push_back(rewriter.getNamedAttr(
            "rshift", rewriter.getSI32IntegerAttr(shift)));
        attrs.push_back(rewriter.getNamedAttr(
            "quant_mode",
            tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
        rewriter.replaceOpWithNewOp<tpu::RequantIntOp>(
            op, op.getOutput().getType(), ValueRange{pre_cast_in}, attrs);
        return success();
      }
    }
    castInputOp.dump();
    op.dump();
    llvm::errs() << "Warning: two cast can merge to one !!!\n";
    return failure();
  }
  bool shouldPrint(tpu::CastOp op) const override { return false; }
};

void tpu::CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<SimplifyRedundantCast>(context);
}

} // namespace tpu
} // namespace tpu_mlir
