//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace llvm;

llvm::SmallVector<int64_t>
conv2d_shape_inference(tpu::Conv2DOp op,
                       llvm::SmallVector<int64_t> input_shape) {
  auto filter_shape = module::getShape(op.getFilter());
  assert(input_shape.size() == filter_shape.size());
  assert(input_shape.size() > 2);
  int spacial_rank = input_shape.size() - 2;
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(filter_shape[0]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  llvm::SmallVector<int64_t> pads({0, 0, 0, 0});
  auto strides = module::getI64Array(op.getStrides());
  llvm::SmallVector<int64_t> dilation({1, 1});
  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] + pads[i] + pads[i + spacial_rank] -
                    dilation[i] * (filter_spacial_shape[i] - 1) - 1) /
                       strides->at(i) +
                   1;
    out_shape.push_back(out_dim);
  }
  return out_shape;
}

namespace tpu_mlir {

namespace bm1684 {
class CastWithoutScalePattern : public OpRewriterPatternEx<tpu::CastOp> {
public:
  CastWithoutScalePattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::CastOp>(context, "CastWithoutScalePattern",
                                         benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::CastOp op,
                      mlir::PatternRewriter &rewriter) const override {
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
      rewriter.eraseOp(op);
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
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

class LargeDilationConvPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  LargeDilationConvPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "LargeDilationConvPattern",
                                           benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::Conv2DOp op,
                      mlir::PatternRewriter &rewriter) const override {
    if (!module::isBM1684Family()) {
      return failure();
    }
    auto strides_v = module::getI64Array(op.getStrides());
    if (1 != strides_v->at(0) || 1 != strides_v->at(1)) {
      return failure();
    }
    auto dilation_v = module::getI64Array(op.getDilations(), 2, 1);
    auto dilation_h = dilation_v->at(0);
    auto dilation_w = dilation_v->at(1);
    if (1 == dilation_h && 1 == dilation_w) {
      return failure();
    }
    auto kernel = module::getI64Array(op.getKernelShape());
    auto kernel_h = kernel->at(0);
    auto kernel_w = kernel->at(1);
    const int64_t convert_thresh = 32;
    if (dilation_h * (kernel_h - 1) + 1 < convert_thresh)
      return failure();
    if (dilation_w * (kernel_w - 1) + 1 < convert_thresh)
      return failure();
    auto pads_v = module::getI64Array(op.getPads());
    auto pad_top = pads_v->at(0);
    auto pad_left = pads_v->at(1);
    auto pad_bottom = pads_v->at(2);
    auto pad_right = pads_v->at(3);
    Value input_value = op->getOperand(0);
    std::string output_name = module::getName(op->getResult(0)).str();
    auto input_shape = module::getShape(input_value);
    auto output_shape = module::getShape(op->getResult(0));
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      return failure();
    }
    auto input_ele_type = module::getElementType(input_value);
    auto output_ele_type = module::getElementType(op.getOutput());

    // 1. Space2BatchOp
    std::string name_space2batch = output_name + "_space2batch";
    auto loc_space2batch =
        NameLoc::get(rewriter.getStringAttr(name_space2batch));
    std::vector<Value> operands_space2batch;
    operands_space2batch.push_back(input_value);
    operands_space2batch.push_back(
        module::getNoneOp(op)); // noneop for bufferop
    std::vector<NamedAttribute> attrs_space2batch;
    auto h = input_shape[2] + pad_top + pad_bottom;
    auto w = input_shape[3] + pad_left + pad_right;
    auto remain_pad_h = align_up(h, dilation_h) - h;
    auto remain_pad_w = align_up(w, dilation_w) - w;
    attrs_space2batch.push_back(rewriter.getNamedAttr(
        "block_h", rewriter.getI64IntegerAttr(dilation_h)));
    attrs_space2batch.push_back(rewriter.getNamedAttr(
        "block_w", rewriter.getI64IntegerAttr(dilation_w)));
    auto pads_attr_space2batch =
        llvm::SmallVector<int64_t>({pad_top, pad_bottom + remain_pad_h,
                                    pad_left, pad_right + remain_pad_w});
    attrs_space2batch.push_back(rewriter.getNamedAttr(
        "pads", rewriter.getI64ArrayAttr(pads_attr_space2batch)));
    auto output_shape_space2batch = llvm::SmallVector<int64_t>(input_shape);
    output_shape_space2batch[2] +=
        pads_attr_space2batch[0] + pads_attr_space2batch[1];
    output_shape_space2batch[3] +=
        pads_attr_space2batch[2] + pads_attr_space2batch[3];
    if (0 != output_shape_space2batch[2] % dilation_h ||
        0 != output_shape_space2batch[3] % dilation_w) {
      return failure();
    }
    output_shape_space2batch[0] *= (dilation_h * dilation_w);
    output_shape_space2batch[2] /= dilation_h;
    output_shape_space2batch[3] /= dilation_w;
    auto op_space2batch = rewriter.create<tpu::Space2BatchOp>(
        loc_space2batch,
        RankedTensorType::get(output_shape_space2batch, input_ele_type),
        operands_space2batch, attrs_space2batch);
    auto output_value_space2batch = op_space2batch.getResult();

    // 2. ConvOp
    // rewriter.setInsertionPointAfterValue(output_value_space2batch);
    std::string name_conv = output_name + "_conv";
    auto loc_conv = NameLoc::get(rewriter.getStringAttr(name_conv));
    std::vector<Value> operands_conv;
    operands_conv.push_back(output_value_space2batch);
    operands_conv.push_back(op.getFilter());
    operands_conv.push_back(op.getBias());
    llvm::SmallVector<int64_t> output_shape_conv =
        conv2d_shape_inference(op, output_shape_space2batch);
    std::vector<NamedAttribute> attrs_conv;
    for (auto &attr : op->getAttrs()) {
      if (attr.getName() == "dilations") {
        attrs_conv.push_back(rewriter.getNamedAttr(
            "dilations", rewriter.getI64ArrayAttr({1, 1})));
      } else if (attr.getName() == "pads") {
        attrs_conv.push_back(rewriter.getNamedAttr(
            "pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
      } else {
        attrs_conv.push_back(attr);
      }
    }
    auto op_conv = rewriter.create<tpu::Conv2DOp>(
        loc_conv, RankedTensorType::get(output_shape_conv, output_ele_type),
        operands_conv, attrs_conv);
    auto output_value_conv = op_conv.getResult();

    // 3. Batch2SpaceOp
    // rewriter.setInsertionPointAfterValue(output_value_conv);
    std::vector<Value> operands_batch2space;
    operands_batch2space.push_back(output_value_conv);
    operands_batch2space.push_back(
        module::getNoneOp(op)); // noneop for bufferop
    std::vector<NamedAttribute> attrs_batch2space;
    int64_t crop_h = dilation_h * output_shape_conv[2] - output_shape[2];
    int64_t crop_w = dilation_w * output_shape_conv[3] - output_shape[3];
    attrs_batch2space.push_back(rewriter.getNamedAttr(
        "block_h", rewriter.getI64IntegerAttr(dilation_h)));
    attrs_batch2space.push_back(rewriter.getNamedAttr(
        "block_w", rewriter.getI64IntegerAttr(dilation_w)));
    auto pads_attr_batch2space =
        llvm::SmallVector<int64_t>({0, crop_h, 0, crop_w});
    attrs_batch2space.push_back(rewriter.getNamedAttr(
        "crops", rewriter.getI64ArrayAttr(pads_attr_batch2space)));
    auto op_batch2space = rewriter.create<tpu::Batch2SpaceOp>(
        op.getLoc(), RankedTensorType::get(output_shape, output_ele_type),
        operands_batch2space, attrs_batch2space);
    rewriter.replaceOp(op, op_batch2space);

    return failure();
  }
};

class Use3icPadConvPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  Use3icPadConvPattern(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "Use3icPadConvPattern",
                                           benifit) {}

protected:
  LogicalResult
  matchAndRewriteImpl(tpu::Conv2DOp op,
                      mlir::PatternRewriter &rewriter) const override {
    auto prevOp = op->getOperand(0).getDefiningOp();
    auto prevInputOp = prevOp;
    if (!isa<top::InputOp>(prevOp)) {
      // prevInputOp = prevOp->getOperand(0).getDefiningOp();
      if (isa<tpu::CastOp>(prevOp) && prevOp->hasOneUse()) {
        prevInputOp = prevOp->getOperand(0).getDefiningOp();
      } else {
        return failure();
      }
    }
    if (isa<top::InputOp>(prevOp) || isa<top::InputOp>(prevInputOp)) {
      if (op->use_empty())
        return failure();
      auto in_type = module::getStorageType(op.getInput());
      if (!(module::isBM1684Family()) || in_type.isF32())
        return failure();

      auto pads_v = module::getI64Array(op.getPads());
      auto pad_top = pads_v->at(0);
      auto pad_left = pads_v->at(1);
      auto pad_bottom = pads_v->at(2);
      auto pad_right = pads_v->at(3);
      bool flag =
          pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0;
      int use_3ic_optimize = 0;
      int use_winograd = 0;
      auto dhdw = module::getI64Array(op.getDilations(), 2, 1);
      auto dh = dhdw->at(0);
      auto i_s = op.getInput().getType().cast<RankedTensorType>().getShape();
      auto ic = i_s[1];
      auto kernel = module::getI64Array(op.getKernelShape());
      auto kh = kernel->at(0);
      auto groups = op.getGroup();
      if (ic == 3 && groups == 1 && dh == 1 && kh > 1 && use_winograd == 0) {
        use_3ic_optimize = 1;
      }
      op->setAttr("use_3ic_optimize",
                  rewriter.getI64IntegerAttr(use_3ic_optimize));
      if (!op.getUse_3icOptimize() || flag) {
        return failure();
      }
      Value input_value = op->getOperand(0);
      std::string output_name = module::getName(op->getResult(0)).str();
      auto input_ele_type = module::getElementType(input_value);
      std::string name_pad = output_name + "_pad";
      auto loc_pad = NameLoc::get(rewriter.getStringAttr(name_pad));
      std::vector<Value> operands_pad;
      operands_pad.push_back(input_value);
      operands_pad.push_back(module::getNoneOp(op));
      operands_pad.push_back(module::getNoneOp(op));
      operands_pad.push_back(module::getNoneOp(op));
      operands_pad.push_back(module::getNoneOp(op));
      llvm::SmallVector<int64_t> pad_paddings(8, 0);
      pad_paddings[2] = pad_top;
      pad_paddings[6] = pad_bottom;
      std::vector<NamedAttribute> attrs_pad;
      attrs_pad.push_back(rewriter.getNamedAttr(
          "paddings", rewriter.getI64ArrayAttr(pad_paddings)));
      attrs_pad.push_back(rewriter.getNamedAttr(
          "mode",
          tpu::PaddingModeAttr::get(getContext(), tpu::PaddingMode::constant)));
      auto input_shape = module::getShape(input_value);
      auto output_shape_pad = llvm::SmallVector<int64_t>(input_shape);
      output_shape_pad[2] += (pad_paddings[2] + pad_paddings[6]);
      auto op_pad = rewriter.create<tpu::PadOp>(
          loc_pad, RankedTensorType::get(output_shape_pad, input_ele_type),
          operands_pad, attrs_pad);
      input_value = op_pad.getResult();
      op.setOperand(0, input_value);
      llvm::SmallVector<int64_t> conv_paddings = {0, pad_left, 0, pad_right};
      op.setPadsAttr(rewriter.getI64ArrayAttr(conv_paddings));
      return success();
    }
    return failure();
  }
};
} // namespace bm1684

namespace tpu {
using namespace bm1684;
void populateOptimizeBM1684Patterns(RewritePatternSet *patterns) {
  auto ctx = patterns->getContext();
  patterns->add<LargePadConvPattern>(ctx, 9);
  patterns->add<CastWithoutScalePattern, LargeDilationConvPattern,
                PermuteReorderPattern, PermutePadSwap, Use3icPadConvPattern,
                RemoveReshape>(ctx, 8);
};
} // namespace tpu

} // namespace tpu_mlir
