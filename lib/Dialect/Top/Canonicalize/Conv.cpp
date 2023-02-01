//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace tpu_mlir::top;

struct Conv3dTo2d : public OpRewritePattern<ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
    auto p = op.parseParam();
    if (op.getKernelShape().size() != 3 || p.id != p.kd) {
      return failure();
    }
    auto in = op.getInput();
    auto out = op.getOutput();
    // in reshape to 4dim
    std::vector<int64_t> in_shape = {p.n, p.ic * p.id, p.ih, p.iw};
    auto newType = RankedTensorType::get(in_shape, module::getElementType(in));
    std::string in_name = module::getName(in).str() + "_To4Dim";
    auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
    rewriter.setInsertionPoint(op);
    auto rs1_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{in});
    op.setOperand(0, rs1_op.getOutput());
    // out reshape to 5dim
    auto outType = out.getType();
    std::string out_name = module::getName(in).str() + "_To5Dim";
    loc = NameLoc::get(rewriter.getStringAttr(out_name));
    rewriter.setInsertionPointAfter(op);
    auto rs2_op = rewriter.create<ReshapeOp>(loc, outType, ValueRange{out});
    out.replaceAllUsesExcept(rs2_op.getOutput(), rs2_op);
    // conv 5d to 4d
    newType = RankedTensorType::get({p.n, p.oc * p.od, p.oh, p.ow},
                                    module::getElementType(out));
    out.setType(newType);
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({p.kh, p.kw}));
    op.setStridesAttr(rewriter.getI64ArrayAttr({p.sh, p.sw}));
    op.setDilationsAttr(rewriter.getI64ArrayAttr({p.dh, p.dw}));
    op.setPadsAttr(rewriter.getI64ArrayAttr({p.pht, p.pwl, p.phb, p.pwr}));
    auto kernel = op.getFilter();
    newType = RankedTensorType::get({p.oc, p.ic * p.kd / p.groups, p.kh, p.kw},
                                    module::getElementType(out));
    kernel.setType(newType);
    return success();
  }
};

void ConvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<Conv3dTo2d>(context);
}
