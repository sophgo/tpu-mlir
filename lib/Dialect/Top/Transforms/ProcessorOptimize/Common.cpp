//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace tpu_mlir {
namespace top {

LogicalResult
MergeScale2Conv::matchAndRewrite(top::ScaleOp op,
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
  std::vector<float_t> scaleVec(c, 1);
  if (scale) {
    auto scaleShape = module::getShape(scale);
    auto scaleData = scale.read<float>();
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

    auto filterData = filterOp.read<float>();
    std::vector<float_t> newFilter(filterData->size(), 0);
    uint32_t innerSize = filterData->size() / c;
    for (uint32_t i = 0; i < c; ++i) {
      for (uint32_t j = 0; j < innerSize; ++j) {
        newFilter.at(i * innerSize + j) =
            filterData->at(i * innerSize + j) * scaleVec.at(i);
      }
    }
    filterOp.update(newFilter, newFilter.size());
  }
  if (sBias) {
    // merge SBias into conv's bias
    auto sBiasShape = module::getShape(sBias);
    auto sBiasData = sBias.read<float>();
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
      auto cBiasData = cBiasOp.read<float>();
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

LogicalResult ConvertScaleOp::matchAndRewrite(top::ScaleOp op,
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
  auto cur_scale_f32 = cur_scale.read<float>();
  auto cur_bias_f32 = cur_bias.read<float>();

  std::vector<float> new_scale_v(channel);
  std::vector<float> new_bias_v(channel);
  std::copy(cur_scale_f32->begin(), cur_scale_f32->end(), new_scale_v.begin());
  std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

  // scale to depthwise convolution
  NamedAttrList attrs;
  attrs.set("kernel_shape", rewriter.getI64ArrayAttr({1, 1}));
  attrs.set("strides", rewriter.getI64ArrayAttr({1, 1}));
  attrs.set("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0}));
  attrs.set("group", rewriter.getI64IntegerAttr(channel));
  attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
  auto relu_limit = op.getReluLimit().convertToDouble();
  attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));

  auto filter_type =
      RankedTensorType::get({channel, 1, 1, 1}, rewriter.getF32Type());
  auto new_scale =
      top::WeightOp::create(op, "to_weight", new_scale_v, filter_type);
  auto bias_type = RankedTensorType::get({channel}, rewriter.getF32Type());
  auto new_bias = top::WeightOp::create(op, "to_bias", new_bias_v, bias_type);

  rewriter.replaceOpWithNewOp<top::ConvOp>(
      op, op.getResult().getType(),
      ValueRange{op.getInput(), new_scale, new_bias}, attrs);
  return success();
}

LogicalResult ConvertScaleToMAOp::matchAndRewrite(top::ScaleOp op,
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
  auto cur_scale_f32 = cur_scale.read<float>();
  auto cur_bias_f32 = cur_bias.read<float>();

  std::vector<float> new_scale_v(channel);
  std::vector<float> new_bias_v(channel);
  std::copy(cur_scale_f32->begin(), cur_scale_f32->end(), new_scale_v.begin());
  std::copy(cur_bias_f32->begin(), cur_bias_f32->end(), new_bias_v.begin());

  // scale to mul and add, for temp
  NamedAttrList attrs;
  attrs.set("do_relu", rewriter.getBoolAttr(op.getDoRelu()));
  auto relu_limit = op.getReluLimit().convertToDouble();
  attrs.set("relu_limit", rewriter.getF64FloatAttr(relu_limit));


  auto new_shape = std::vector <int64_t>(input_shape.size(), 1);
  new_shape[1] = channel;
  auto filter_type =
      RankedTensorType::get(new_shape, rewriter.getF32Type());
  auto new_scale =
      top::WeightOp::create(op, "to_weight", new_scale_v, filter_type);

  std::string new_name = module::getName(op.getOperation()).str() + "_bias";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
  auto new_m = rewriter.create<top::MulOp>(name_loc, op.getOutput().getType(), ValueRange{op.getInput(), new_scale}, attrs);

  new_m.getOperation()->setAttr("do_relu", rewriter.getBoolAttr(false));
  new_m.getOperation()->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.0));

  auto bias_type = RankedTensorType::get(new_shape, rewriter.getF32Type());
  auto new_bias = top::WeightOp::create(new_m, "to_bias", new_bias_v, bias_type);

  rewriter.setInsertionPointAfterValue(new_m);
  auto new_b = rewriter.replaceOpWithNewOp<top::AddOp>(
      op, new_m.getResult().getType(),
      ValueRange{new_m.getOutput(), new_bias}, attrs);
  new_m.getOutput().replaceAllUsesExcept(new_b.getOutput(), new_b.getOperation());
  return success();
}

// A --slice--> A0 | A1 --concat--> A1 | A0
// ==> SwapDimInner
// test by `test_onnx.py --case SwapDimInner`
LogicalResult ConcatToSwapDimInner::matchAndRewrite(top::ConcatOp concat_op,
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
  if (from != slice1_op.getInput()) {
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

  int axis = concat_op.getAxis();
  if (offset0->at(axis) != oshape1[axis] || offset1->at(axis) != 0) {
    return failure();
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(*offset0)));
  // concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(
  // module::getName(concat_op.getOperation()).str() + "_SwapDimInner")));
  rewriter.replaceOpWithNewOp<top::SwapDimInnerOp>(
      concat_op, concat_op.getResult().getType(), ValueRange{from}, attrs);

  return success();
}


} // namespace top
} // namespace tpu_mlir
