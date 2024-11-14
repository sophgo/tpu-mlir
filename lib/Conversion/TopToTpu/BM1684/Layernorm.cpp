//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void LayerNormLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::LayerNormOp op) const {
  rewriter.setInsertionPointAfter(op);
  /*BM1684 do not support LayerNorm, use GroupNorm instead of LayerNorm, if have
   * weights or bias, use MulOp and AddOp replace*/
  auto weight = op.getWeight();
  auto bias = op.getBias();
  SmallVector<Value, 5> opds;
  opds.push_back(op.getOperand(0));
  // fill weight and bias use NoneOp
  opds.push_back(module::getNoneOp(op));
  opds.push_back(module::getNoneOp(op));
  // table for cv18xx no use
  opds.push_back(module::getNoneOp(op));
  opds.push_back(module::getNoneOp(op));
  auto name = module::getName(op.getOutput());
  auto input_shape = module::getShape(op.getInput());
  bool has_weight = !module::isNone(weight);
  bool has_bias = !module::isNone(bias);
  auto axis = op.getAxis();
  int64_t group_num;
  if (axis == 1) {
    group_num = 1;
  } else if (axis == 2) {
    group_num = input_shape[1];
  } else if (axis == 3) {
    group_num = input_shape[1] * input_shape[2];
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }

  std::vector<NamedAttribute> attrs;
  auto new_type = op.getOutput().getType();
  Value last_op_value = op.getOutput();
  // replace LayerNorm use GroupNorm
  if (axis != 3) {
    attrs.push_back(rewriter.getNamedAttr(
        "num_groups", rewriter.getI64IntegerAttr(group_num)));
    attrs.push_back(rewriter.getNamedAttr("eps", op.getEpsAttr()));
    auto group_norm_loc = (has_weight || has_bias)
                              ? NameLoc::get(rewriter.getStringAttr(
                                    name.str() + "_convert_to_group_norm"))
                              : op.getLoc();
    auto group_norm_op = rewriter.create<tpu::GroupNormOp>(
        group_norm_loc, new_type, opds, attrs);

    last_op_value = group_norm_op.getOutput();
  } else {
    // Handle the case when axis is 3
    SmallVector<Value, 2> opds_1;
    opds_1.push_back(op.getOperand(0));
    opds_1.push_back(module::getNoneOp(op));
    std::vector<int64_t> shape_1 = {input_shape[0], group_num, 1,
                                    input_shape[3]};
    auto type_1 = RankedTensorType::get(shape_1, rewriter.getF32Type());
    auto reshape_1_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape_1"));
    auto reshape_1_op =
        rewriter.create<tpu::ReshapeOp>(reshape_1_loc, type_1, opds_1);

    opds[0] = reshape_1_op.getResult();
    auto group_norm_type = reshape_1_op.getOutput().getType();
    attrs.push_back(rewriter.getNamedAttr(
        "num_groups", rewriter.getI64IntegerAttr(group_num)));
    attrs.push_back(rewriter.getNamedAttr("eps", op.getEpsAttr()));
    auto group_norm_loc = NameLoc::get(
        rewriter.getStringAttr(name.str() + "_convert_to_group_norm"));
    auto group_norm_op = rewriter.create<tpu::GroupNormOp>(
        group_norm_loc, group_norm_type, opds, attrs);

    SmallVector<Value, 2> opds_2;
    opds_2.push_back(group_norm_op.getOutput());
    opds_2.push_back(module::getNoneOp(op));
    auto type_2 = RankedTensorType::get(input_shape, rewriter.getF32Type());
    auto reshape_2_loc =
        (has_weight || has_bias)
            ? NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape_2"))
            : op.getLoc();
    auto reshape_2_op =
        rewriter.create<tpu::ReshapeOp>(reshape_2_loc, type_2, opds_2);

    last_op_value = reshape_2_op.getOutput();
  }

  attrs.clear();
  // replace weight use Mul
  if (has_weight) {
    auto mul_loc =
        has_bias
            ? NameLoc::get(rewriter.getStringAttr(name.str() + "_mul_weights"))
            : op.getLoc();
    auto mul_op = rewriter.create<tpu::MulOp>(
        mul_loc, new_type, ValueRange{last_op_value, weight}, attrs);
    last_op_value = mul_op.getOutput();
  }

  attrs.clear();
  // replace bias use Add
  if (has_bias) {
    auto bias_loc = op.getLoc();
    auto bias_op = rewriter.create<tpu::AddOp>(
        bias_loc, new_type, ValueRange{last_op_value, bias}, attrs);
    last_op_value = bias_op.getOutput();
  }
  op.getResult().replaceAllUsesWith(last_op_value);
  rewriter.eraseOp(op);
}

void LayerNormLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::LayerNormOp op,
                                     bool asymmetric) const {
  // only support FP32
  LoweringF32(rewriter, op);
}

} // namespace bm1684
} // namespace tpu_mlir
