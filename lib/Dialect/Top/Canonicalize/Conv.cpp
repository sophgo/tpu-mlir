//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::top;

struct Conv1dTo2d : public OpRewritePattern<ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {

    auto kernel = module::getI64Array(op.getKernelShape());
    if (kernel->size() != 1) {
      return failure();
    }
    std::vector<int64_t> vfilterShape = module::getShape(op.getFilter());
    vfilterShape.push_back(1);
    auto new_type = RankedTensorType::get(vfilterShape, rewriter.getF32Type());
    op.getFilter().setType(new_type);

    // update kernel_shape
    std::vector<int64_t> kernel_shape =
        *module::getI64Array(op.getKernelShape());
    kernel_shape.push_back(1);
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr(kernel_shape));
    std::vector<int64_t> strides = *module::getI64Array(op.getStrides());
    strides.push_back(1);
    op.setStridesAttr(rewriter.getI64ArrayAttr(strides));
    // update pads
    auto pads_v = module::getI64Array(op.getPads());
    std::vector<int64_t> pads = {pads_v->at(0), 0, pads_v->at(1), 0};
    op.setPadsAttr(rewriter.getI64ArrayAttr(pads));
    // update dilations
    std::vector<int64_t> dilations =
        *module::getI64Array(op.getDilations(), 1, 1);
    dilations.push_back(1);
    op.setDilationsAttr(rewriter.getI64ArrayAttr(dilations));
    return success();
  }
};

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
struct Conv3dTranspose : public OpRewritePattern<ConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op,
                                PatternRewriter &rewriter) const override {
    /* make sure it is a Conv3dOp and id == kd and ic == filter ic */
    auto p = op.parseParam();
    auto filter = op.getFilter();
    auto f_shape = module::getShape(filter);
    if (op.getKernelShape().size() != 3 || p.id != p.kd || f_shape[1] != p.ic) {
      return failure();
    }
    /* make sure the input comes from PermuteOp */
    auto in = op.getInput();
    auto tp = dyn_cast<PermuteOp>(in.getDefiningOp());
    if (!tp) {
      return failure();
    }
    /* make sure the input is the only output of PermuteOp */
    if (!tp.getOutput().hasOneUse()) {
      return failure();
    }
    /* make sure the PermuteOp is between dim 1 and 2 */
    std::vector<int64_t> ps = {0, 2, 1, 3, 4};
    auto order = module::getI64Array(tp.getOrder());
    if (*order != ps) {
      return failure();
    }
    /* transpose the filter */
    auto filter_op = filter.getDefiningOp<top::WeightOp>();
    auto filter_data = filter_op.read<float>();
    auto filter_tp =
        std::make_shared<std::vector<float>>(filter_data->size(), 0);
    function_permute(filter_data->data(), filter_tp->data(), f_shape, ps);
    std::vector<int64_t> f_shape_tp = {f_shape[0], f_shape[2], f_shape[1],
                                       f_shape[3], f_shape[4]};
    /* get rid of PermuteOp */
    tp.getOutput().replaceAllUsesWith(
        tp.getInput()); // this replaces op.getInput() with tp.getInput().
    rewriter.eraseOp(tp);
    /* create a new weight for the transposed filter */
    rewriter.setInsertionPointAfter(op);
    auto type = RankedTensorType::get(f_shape_tp, rewriter.getF32Type());
    auto weight =
        WeightOp::create<float>(op, "transposed_weight", *filter_tp,
                                type); // this is Weight itself, not WeightOp
    /* change the attr of conv3d op */
    op.setOperand(
        1,
        weight); // op.setOperand vs op->setOperand: in this case both OK. This
                 // replaces op.getFilter() with the transposed filter $weight.
    rewriter.eraseOp(filter_op); // remove unused WeightOp manually, optional
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({p.ic, p.kh, p.kw}));
    return success();
  }
};

void ConvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<Conv3dTranspose, Conv3dTo2d, Conv1dTo2d>(context);
}
