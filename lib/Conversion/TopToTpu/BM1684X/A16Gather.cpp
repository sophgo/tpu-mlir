//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void A16GatherLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::A16GatherOp op) const {
  llvm_unreachable("Not implement");
}

void A16GatherLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::A16GatherOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void A16GatherLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::A16GatherOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void A16GatherLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::A16GatherOp op) const {
  auto newType = getQuantBF16Type(op->getResult(0));
  std::vector<Value> operands;

  // weight: keep as uint8
  auto weight_value = op.getWeight();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  auto weight_data = weight_op.read<uint8_t>();
  auto new_weight_type = RankedTensorType::get(
      weight_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_weight_value =
      top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
  operands.push_back(new_weight_value);

  // indices: keep as-is
  operands.push_back(op->getOperand(1));

  // scale: convert to bf16, or pass NoneOp through
  auto scale_value = op.getScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  if (scale_op) {
    auto new_scale_value = scale_op.clone_bf16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }

  // zp: keep as uint8, or pass NoneOp through
  auto zp_value = op.getZp();
  auto zp_op = dyn_cast<top::WeightOp>(zp_value.getDefiningOp());
  if (zp_op) {
    auto zp_data = zp_op.read<uint8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  attrs.push_back(rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
  attrs.push_back(rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr()));

  rewriter.replaceOpWithNewOp<tpu::A16GatherOp>(op, newType, operands, attrs);
}

void A16GatherLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::A16GatherOp op) const {
  auto newType = getQuantF16Type(op->getResult(0));
  std::vector<Value> operands;

  // weight: keep as uint8
  auto weight_value = op.getWeight();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  auto weight_data = weight_op.read<uint8_t>();
  auto new_weight_type = RankedTensorType::get(
      weight_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_weight_value =
      top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
  operands.push_back(new_weight_value);

  // indices: keep as-is
  operands.push_back(op->getOperand(1));

  // scale: convert to f16, or pass NoneOp through
  auto scale_value = op.getScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  if (scale_op) {
    auto new_scale_value = scale_op.clone_f16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }

  // zp: keep as uint8, or pass NoneOp through
  auto zp_value = op.getZp();
  auto zp_op = dyn_cast<top::WeightOp>(zp_value.getDefiningOp());
  if (zp_op) {
    auto zp_data = zp_op.read<uint8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
  attrs.push_back(rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
  attrs.push_back(rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr()));

  rewriter.replaceOpWithNewOp<tpu::A16GatherOp>(op, newType, operands, attrs);
}

void A16GatherLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::A16GatherOp op) const {
  llvm_unreachable("Not implement");
}

void A16GatherLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::A16GatherOp op) const {
  llvm_unreachable("Not implement");
}

} // namespace bm1684x
} // namespace tpu_mlir
