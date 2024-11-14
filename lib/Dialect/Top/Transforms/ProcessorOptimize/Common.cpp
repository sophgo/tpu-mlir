//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Support/Float16.h"
#include <cmath>

namespace tpu_mlir {
namespace top {

LogicalResult
MergeScale2Conv::matchAndRewriteImpl(top::ScaleOp op,
                                     PatternRewriter &rewriter) const {
  auto preOp = op.getInput().getDefiningOp();
  if (!preOp->hasOneUse() || !isa<ConvOp>(preOp)) {
    return failure();
  }
  auto convOp = cast<ConvOp>(preOp);
  if (convOp.getDoRelu()) {
    return failure();
  }
  auto c = module::getShape(convOp.getOutput())[1];
  auto scale = dyn_cast<WeightOp>(op.getScale().getDefiningOp());
  auto sBias = dyn_cast<WeightOp>(op.getBias().getDefiningOp());
  if (!sBias) {
    return failure();
  }
  auto cur_storage_type = module::getStorageType(op.getOutput());
  if (!cur_storage_type.isF32() && !cur_storage_type.isF16()) {
    return failure();
  }
  auto pre_storage_type = module::getStorageType(convOp.getOutput());
  if (cur_storage_type != pre_storage_type) {
    return failure();
  }

  std::vector<float_t> scaleVec(c, 1);
  if (scale) {
    auto scaleShape = module::getShape(scale);
    auto scaleData = scale.read_as_float();
    scaleVec.assign(scaleData->begin(), scaleData->end());
    if (std::find(scaleShape.begin(), scaleShape.end(), c) ==
            scaleShape.end() &&
        scaleVec.size() != c) {
      return failure();
    }
    auto filterOp = dyn_cast<WeightOp>(convOp.getFilter().getDefiningOp());
    if (!filterOp) { // filter may be not WeightOp
      return failure();
    }

    auto filterData = filterOp.read_as_float();
    std::vector<float_t> newFilter(filterData->size(), 0);
    uint32_t innerSize = filterData->size() / c;
    for (uint32_t i = 0; i < c; ++i) {
      for (uint32_t j = 0; j < innerSize; ++j) {
        newFilter.at(i * innerSize + j) =
            filterData->at(i * innerSize + j) * scaleVec.at(i);
      }
    }
    if (cur_storage_type.isF32()) {
      filterOp.update(newFilter, newFilter.size());
    } else {
      std::vector<uint16_t> newFilterF16(filterData->size(), 0);
#pragma omp parallel for schedule(static, omp_schedule(filterData->size()))
      for (uint64_t i = 0; i < filterData->size(); ++i) {
        newFilterF16.at(i) = f32_to_f16(filterData->at(i));
      }
      filterOp.update(newFilterF16, newFilterF16.size());
    }
  }
  if (sBias) {
    // merge SBias into conv's bias
    auto sBiasShape = module::getShape(sBias);
    auto sBiasData = sBias.read_as_float();
    if (std::find(sBiasShape.begin(), sBiasShape.end(), c) ==
            sBiasShape.end() &&
        sBiasData->size() != c) {
      return failure();
    }
    std::vector<float_t> newBiasVec(c, 0);
    newBiasVec.assign(sBiasData->begin(), sBiasData->end());
    auto newBiasType = RankedTensorType::get({c}, rewriter.getF32Type());
    if (!module::isNone(convOp.getBias())) {
      auto cBiasOp = dyn_cast<WeightOp>(convOp.getBias().getDefiningOp());
      if (!cBiasOp) { // filter may be not WeightOp
        return failure();
      }
      auto cBiasData = cBiasOp.read_as_float();
      for (int i = 0; i < c; ++i) {
        newBiasVec[i] += cBiasData->at(i) * scaleVec[i];
      }
      cBiasOp.update(newBiasVec, c);
    } else {
      auto newBiasOp = WeightOp::create(convOp, module::getName(sBias, 0).str(),
                                        newBiasVec, newBiasType);
      convOp.setOperand(2, newBiasOp);
    }
  }
  // update attrs
  preOp->setLoc(op.getLoc());
  preOp->setAttr("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
  preOp->setAttr("relu_limit",
                 rewriter.getF64FloatAttr(op.getReluLimit().convertToDouble()));
  convOp.getOutput().setType(op.getOutput().getType());
  // remove scale Op
  rewriter.replaceOp(op, {op.getInput()});
  return success();
}

LogicalResult
ConvertScaleOp::matchAndRewriteImpl(top::ScaleOp op,
                                    PatternRewriter &rewriter) const {
  auto input_shape = module::getShape(op.getInput());
  if (input_shape.size() > 4) {
    return failure();
  }
  auto cur_scale = dyn_cast<top::WeightOp>(op.getScale().getDefiningOp());
  auto cur_bias = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
  if (!(cur_scale && cur_bias) || input_shape.size() < 3) {
    return failure();
  }
  int channel = cur_scale.getType().cast<RankedTensorType>().getNumElements();
  auto cur_bias_f32 = cur_bias.read_as_float();

  auto scale_new_type = module::getTypeLike(cur_scale, {channel, 1, 1, 1});
  op.getScale().setType(scale_new_type);

  // scale to depthwise convolution
  NamedAttrList attrs;
  attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
  attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
  attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
  attrs.set("group", rewriter.getI64IntegerAttr(channel));
  attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
  auto relu_limit = op.getReluLimit().convertToDouble();
  attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

  auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
  auto new_bias =
      top::WeightOp::create(op, "to_bias", *cur_bias_f32, bias_type);

  rewriter.replaceOpWithNewOp<top::ConvOp>(
      op, op.getResult().getType(),
      ValueRange{op.getInput(), cur_scale, new_bias}, attrs);
  return success();
}

// A --slice--> A0 | A1 --concat--> A1 | A0
// ==> SwapDimInner
// test by `test_onnx.py --case SwapDimInner`
LogicalResult
ConcatToSwapDimInner::matchAndRewriteImpl(top::ConcatOp concat_op,
                                          PatternRewriter &rewriter) const {
  if (concat_op.getDoRelu()) {
    return failure();
  }

  int num_inputs = concat_op.getInputs().size();
  if (num_inputs != 2) {
    return failure();
  }

  auto in0_op = concat_op.getInputs()[0].getDefiningOp();
  auto in1_op = concat_op.getInputs()[1].getDefiningOp();
  auto slice0_op = dyn_cast<top::SliceOp>(in0_op);
  auto slice1_op = dyn_cast<top::SliceOp>(in1_op);
  if (!slice0_op || !slice1_op) {
    return failure();
  }
  auto from = slice0_op.getInput();
  // ensure slice'ancestor has only two slices
  if (from != slice1_op.getInput() ||
      std::distance(from.user_begin(), from.user_end()) != 2) {
    return failure();
  }
  auto steps0 = module::getI64Array(slice0_op.getSteps());
  auto steps1 = module::getI64Array(slice1_op.getSteps());
  for (int i = 0; i < steps0->size(); ++i) {
    if (steps0->at(i) != 1 || steps0->at(i) != steps1->at(i)) {
      return failure();
    }
  }
  auto offset0 = module::getI64Array(slice0_op.getOffset());
  auto offset1 = module::getI64Array(slice1_op.getOffset());
  // auto oshape0 = module::getShape(slice0_op.getOutput());
  auto oshape1 = module::getShape(slice1_op.getOutput());
  auto fshape = module::getShape(from);
  int axis = concat_op.getAxis();
  int offset_axis0 = offset0->at(axis);
  int offset_axis1 = offset1->at(axis);
  if (offset0->at(axis) < 0) {
    offset_axis0 = fshape[axis] + offset0->at(axis);
    offset0->at(axis) = offset_axis0;
  }
  if (offset1->at(axis) < 0) {
    offset_axis1 = fshape[axis] + offset1->at(axis);
  }
  if (offset_axis0 != oshape1[axis] || offset_axis1 != 0) {
    return failure();
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(*offset0)));
  // concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(
  // module::getName(concat_op.getOperation()).str() + "_SwapDimInner")));
  rewriter.replaceOpWithNewOp<top::SwapDimInnerOp>(
      concat_op, concat_op.getResult().getType(), ValueRange{from}, attrs);
  //         / slice \
// (544,960)         concat(960,544) ... concat(960,544,960,544)
  //         \ slice /
  if (in0_op->getUses().empty())
    rewriter.eraseOp(in0_op);
  if (in1_op->getUses().empty())
    rewriter.eraseOp(in1_op);
  return success();
}
} // namespace top
} // namespace tpu_mlir
