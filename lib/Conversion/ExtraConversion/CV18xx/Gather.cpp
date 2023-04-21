//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/ExtraConversion/ExtraConvertCV18XX.h"

namespace tpu_mlir {
namespace cv18xx {

LogicalResult
ConvertGatherOp::matchAndRewrite(top::GatherOp op,
                                 PatternRewriter &rewriter) const {
  // for transform decode's index op
  Value input = op.getInput();
  Value indices = op.getIndices();
  Value ori_out = op.getOutput();
  std::string name = module::getName(ori_out).str();
  uint64_t axis = op.getAxis();
  std::vector<int64_t> input_shape = module::getShape(input);
  std::vector<int64_t> output_shape = module::getShape(ori_out);
  std::vector<int64_t> indices_shape = module::getShape(indices);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;

  // convert to slice
  auto weight_op = dyn_cast_or_null<top::WeightOp>(indices.getDefiningOp());
  if (weight_op) {
    auto weight_data = weight_op.read_as_float();
    // indices size == 1
    if (weight_data->size() == 1) {
      assert(input_shape.size() == output_shape.size());
      int offset = static_cast<int>(weight_data->at(0));
      if (offset < 0) {
        offset = input_shape[axis] + offset;
      }
      std::vector<int64_t> slice_offsets(input_shape.size(), 0);
      std::vector<int64_t> slice_steps(input_shape.size(), 1);
      slice_offsets[axis] = offset;
      operands.emplace_back(input);
      attrs.emplace_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr(slice_offsets)));
      attrs.emplace_back(rewriter.getNamedAttr(
          "steps", rewriter.getI64ArrayAttr(slice_steps)));
      rewriter.replaceOpWithNewOp<top::SliceOp>(op, ori_out.getType(), operands,
                                                attrs);
      return success();
    } else {
      // indices is an arithmetic progression
      int diff = static_cast<int>(weight_data->at(1)) -
                 static_cast<int>(weight_data->at(0));
      if (diff == 0) {
        return failure();
      }
      int last = static_cast<int>(weight_data->at(1));
      bool is_arithmetic_progression = true;
      for (int i = 2; i < weight_data->size(); ++i) {
        int tmp = static_cast<int>(weight_data->at(i)) - last;
        if (tmp != diff) {
          is_arithmetic_progression = false;
          break;
        }
        last = static_cast<int>(weight_data->at(i));
      }
      if (is_arithmetic_progression) {
        int _axis = axis >= 0 ? axis : axis + input_shape.size();
        std::vector<int64_t> slice_offsets(input_shape.size(), 0);
        std::vector<int64_t> slice_steps(input_shape.size(), 1);
        slice_offsets[_axis] = diff > 0
                                  ? static_cast<int>(*weight_data->begin())
                                  : static_cast<int>(*weight_data->rbegin());
        slice_steps[_axis] = std::abs(diff);
        operands.emplace_back(input);
        attrs.emplace_back(rewriter.getNamedAttr(
            "offset", rewriter.getI64ArrayAttr(slice_offsets)));
        attrs.emplace_back(rewriter.getNamedAttr(
            "steps", rewriter.getI64ArrayAttr(slice_steps)));
        rewriter.replaceOpWithNewOp<top::SliceOp>(op, ori_out.getType(),
                                                  operands, attrs);
        return success();
      }
    }
  } else {
    // convert for embedding
    bool need_convert =
        (axis == 1 && indices_shape.size() == 0 && input_shape.size() == 3 &&
         input_shape[0] == 1 && !(isa<top::WeightOp>(input.getDefiningOp())));
    if (need_convert) {
      // conver to reshapeOp + new GatherOp
      rewriter.setInsertionPointAfterValue(ori_out);
      double in_thr, out_thr;
      RankedTensorType type1, type2;
      if (module::isCalibratedType(ori_out)) {
        auto itype = module::getCalibratedType(input);
        auto otype = module::getCalibratedType(ori_out);
        auto caliType1 = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -in_thr, in_thr);
        auto caliType2 = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -out_thr, out_thr);
        type1 =
            RankedTensorType::get({input_shape[1], input_shape[2]}, caliType1);
        type2 = RankedTensorType::get(output_shape, caliType2);
      } else {
        type1 = RankedTensorType::get({input_shape[1], input_shape[2]},
                                      rewriter.getF32Type());
        type2 = ori_out.getType().cast<RankedTensorType>();
      }
      operands.emplace_back(input);
      auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
      auto reshapeOp =
          rewriter.create<top::ReshapeOp>(loc1, type1, operands, attrs);
      auto out1 = reshapeOp.getOutput();
      operands.clear();
      operands.emplace_back(out1);
      operands.emplace_back(indices);
      attrs.emplace_back(
          rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(0)));
      auto loc2 = NameLoc::get(rewriter.getStringAttr(name));
      auto newOp = rewriter.create<top::GatherOp>(loc2, type2, operands, attrs);
      auto newOut = newOp.getOutput();
      rewriter.replaceAllUsesWith(ori_out, newOut);
      rewriter.eraseOp(op);
      return success();
    }
  }
  return failure();
}
} // namespace cv18xx
} // namespace tpu_mlir
