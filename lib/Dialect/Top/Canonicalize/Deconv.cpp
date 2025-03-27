//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

struct ReorderDynWeight : public OpRewriterPatternEx<DeconvOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ReorderDynWeight(mlir::MLIRContext *context)
      : OpRewriterPatternEx<DeconvOp>(context, "ReorderDynWeight") {}

  LogicalResult matchAndRewriteImpl(DeconvOp op,
                                    PatternRewriter &rewriter) const override {

    auto filter_shape =
        module::getShape(op.getFilter()); // <oc, ic, *, *> or <ic, oc, *, *>
    //  first channle of filter should be oc
    // auto out_shape = module::getShape(op.getOutput()); // <bs, oc, *, *>

    // op.getDynWeightReordered()

    if (module::isWeight(op.getOperand(1))) {
      return failure();
    }
    bool dyn_weight_reorderd = op.getDynweightReorderd();
    if (dyn_weight_reorderd) {
      return failure();
    }

    if (isa<top::PermuteOp>(op.getOperand(1).getDefiningOp())) {
      auto permute_op =
          dyn_cast<top::PermuteOp>(op.getOperand(1).getDefiningOp());

      // erase if already have this permute but from original graph
      std::vector<int64_t> ps = {1, 0, 2, 3};
      auto order = module::getI64Array(permute_op.getOrder());
      if (*order == ps) {
        permute_op.replaceAllUsesWith(permute_op.getInput());
        rewriter.eraseOp(permute_op);
        op.setDynweightReorderd(true);
        return success();
      }
    }

    rewriter.setInsertionPointAfterValue(op.getFilter());
    std::string name = module::getName(op.getOutput()).str();
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_reorder_permute"));

    std::vector<int64_t> order = {1, 0};
    auto filter_dim = filter_shape.size();
    for (int i = 2; i < filter_dim; i++) {
      order.push_back(i);
    }

    auto p_type =
        UnrankedTensorType::get(module::getElementType(op.getFilter()));
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));

    auto new_permute_op = rewriter.create<top::PermuteOp>(
        loc, p_type, ValueRange{op.getFilter()}, attrs);

    new_permute_op.shape_inference();
    op.setOperand(1, new_permute_op.getOutput());
    op.setDynweightReorderd(true);
    return success();
  }
};

struct Deconv1dTo2d : public OpRewriterPatternEx<DeconvOp> {
public:
  Deconv1dTo2d(mlir::MLIRContext *context)
      : OpRewriterPatternEx<DeconvOp>(context, "Deconv1dTo2d") {}

  LogicalResult matchAndRewriteImpl(DeconvOp op,
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
    // update output_padding
    std::vector<int64_t> output_padding =
        *module::getI64Array(op.getOutputPadding(), 1, 0);
    output_padding.push_back(0);
    op.setOutputPaddingAttr(rewriter.getI64ArrayAttr(output_padding));
    return success();
  }
};

struct CastFilterAndOutput : public OpRewriterPatternEx<DeconvOp> {
public:
  CastFilterAndOutput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<DeconvOp>(context, "CastFilterAndOutput") {}

  LogicalResult matchAndRewriteImpl(DeconvOp op,
                                    PatternRewriter &rewriter) const override {
    auto input_type = op.getInput().getType().cast<RankedTensorType>();
    if (!input_type.getElementType().isF32()) {
      return failure();
    }
    auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
    if (!filter_type.getElementType().isF16()) {
      return failure();
    }
    auto input_op = op.getOperand(0);
    auto bias_op = op.getOperand(2);
    auto filter_op = op.getOperand(1);
    std::string op_name = module::getName(op.getOutput()).str();
    auto F32Type = module::getElementType(input_op);
    auto filter_shape = module::getShape(filter_op);
    auto new_filter_type = RankedTensorType::get({filter_shape}, F32Type);
    auto F16Type = module::getElementType(filter_op);
    auto out_shape = module::getShape(op.getOutput());
    auto f32_output_type = RankedTensorType::get({out_shape}, F32Type);
    auto f16_output_type = RankedTensorType::get({out_shape}, F16Type);
    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    operands.push_back(filter_op);
    attrs.push_back(rewriter.getNamedAttr(
        "round_mode", rewriter.getStringAttr("HalfAwayFromZero")));
    auto filter_loc =
        NameLoc::get(rewriter.getStringAttr(op_name + "_cast_filter_f32"));
    auto cast_filter_op = rewriter.create<top::CastOp>(
        filter_loc, new_filter_type, operands, attrs);
    attrs.clear();
    operands.clear();
    operands.push_back(input_op);
    operands.push_back(cast_filter_op);
    operands.push_back(bias_op);
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    auto new_loc =
        NameLoc::get(rewriter.getStringAttr(op_name + "_r_cast_output_f32"));
    auto cast_deconv_op = rewriter.create<top::DeconvOp>(
        new_loc, f32_output_type, operands, attrs);
    attrs.clear();
    operands.clear();
    auto f32_output_op = cast_deconv_op.getOutput();
    operands.push_back(f32_output_op);
    attrs.push_back(rewriter.getNamedAttr(
        "round_mode", rewriter.getStringAttr("HalfAwayFromZero")));
    auto cast_loc = NameLoc::get(rewriter.getStringAttr(op_name));
    auto cast_output_op = rewriter.create<top::CastOp>(
        cast_loc, f16_output_type, operands, attrs);
    rewriter.replaceAllUsesWith(op, cast_output_op->getResult(0));
    rewriter.eraseOp(op);
    return failure();
  }
};
void DeconvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<Deconv1dTo2d, ReorderDynWeight, CastFilterAndOutput>(context);
}
