//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"
#include "tpu_mlir/Support/Module.h"

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
  auto none_op = module::getNoneOp(op);
  opds.push_back(module::getNoneOp(op));
  opds.push_back(module::getNoneOp(op));
  auto name = module::getName(op.getOutput());
  bool has_weight = !module::isNone(weight);
  bool has_bias = !module::isNone(bias);
  auto axis = op.getAxis();
  int64_t group_num;
  if (axis == 1) {
    group_num = 1;
  } else if (axis == 2) {
    group_num = module::getShape(op.getInput())[1];
  } else {
    llvm_unreachable("Not Implemented");
  }

  std::vector<NamedAttribute> attrs;
  // replace LayerNorm use GroupNorm
  attrs.push_back(
      rewriter.getNamedAttr("num_groups", rewriter.getI64IntegerAttr(group_num)));
  attrs.push_back(rewriter.getNamedAttr("eps", op.getEpsAttr()));
  auto new_type = op.getOutput().getType();
  auto group_norm_loc = (has_weight || has_bias)
                            ? NameLoc::get(rewriter.getStringAttr(
                                  name.str() + "_convert_to_group_norm"))
                            : op.getLoc();
  auto group_norm_op =
      rewriter.create<tpu::GroupNormOp>(group_norm_loc, new_type, opds, attrs);

  attrs.clear();
  auto last_op_value = group_norm_op.getOutput();
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
