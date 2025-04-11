//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/Float8.h"
namespace tpu_mlir {
namespace bm1684x {

void transform_from_int32_to_int8(
    int length, std::shared_ptr<std::vector<int32_t>> &int32_data,
    std::shared_ptr<std::vector<int8_t>> &int8_data) {
  for (int index = 0; index < length * 4; index++) {
    int32_t value = int32_data->at(index / 4);
    for (int i = 0; i < 4; i++) {
      int8_t int4_1 = static_cast<int8_t>((value >> (i * 8)) & 0xF);
      int8_t int4_2 = static_cast<int8_t>((value >> (i * 8 + 4)) & 0xF);
      int8_t int8_value = static_cast<int8_t>((int4_2 << 4) | int4_1);
      int8_data->at(index + i) = int8_value;
    }
    index += 3;
  }
}

void A16MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::A16MatMulOp op) const {
  llvm_unreachable("Not implement");
}

void A16MatMulLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::A16MatMulOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void A16MatMulLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::A16MatMulOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void A16MatMulLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::A16MatMulOp op) const {
  auto newType = getQuantBF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));

  // lowering weights
  auto weight_value = op.getWeight();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  auto weight_data = weight_op.read<int8_t>();
  auto new_weight_type = RankedTensorType::get(
      weight_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_weight_value =
      top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
  operands.push_back(new_weight_value);

  // lowering scales
  auto scale_value = op.getScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  auto new_scale_value = scale_op.clone_bf16(op);
  operands.push_back(new_scale_value);

  // lowering zps
  auto zp_op = op.getZp().getDefiningOp<top::WeightOp>();
  auto zp_data = zp_op.read<int8_t>();
  auto new_zp_type = RankedTensorType::get(
      // {N, K * 8 / (int)q_group_size},
      zp_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                          .getDefiningOp<top::WeightOp>();
  operands.push_back(new_zp_value);

  // lowering bias
  auto bias_value = op.getBias();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_bf16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  std::vector<NamedAttribute> attrs;
  auto weight_bits_attr =
      rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
  attrs.push_back(weight_bits_attr);

  auto w_transpose_attr =
      rewriter.getNamedAttr("w_transpose", op.getRightTransposeAttr());
  attrs.push_back(w_transpose_attr);

  auto q_group_size_attr =
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr());
  attrs.push_back(q_group_size_attr);

  rewriter.replaceOpWithNewOp<tpu::A16MatMulOp>(op, newType, operands, attrs);
}

void A16MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::A16MatMulOp op) const {
  auto newType = getQuantF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));

  // lowering weights
  auto weight_value = op.getWeight();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  auto weight_data = weight_op.read<int8_t>();
  auto new_weight_type = RankedTensorType::get(
      weight_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_weight_value =
      top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
  operands.push_back(new_weight_value);

  // lowering scales
  auto scale_value = op.getScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  auto new_scale_value = scale_op.clone_f16(op);
  operands.push_back(new_scale_value);

  // lowering zps
  auto zp_op = op.getZp().getDefiningOp<top::WeightOp>();
  auto zp_data = zp_op.read<int8_t>();
  auto new_zp_type = RankedTensorType::get(
      // {N, K * 8 / (int)q_group_size},
      zp_op.getType().cast<RankedTensorType>().getShape(),
      rewriter.getIntegerType(8, false));
  auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                          .getDefiningOp<top::WeightOp>();
  operands.push_back(new_zp_value);

  // lowering bias
  auto bias_value = op.getBias();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_f16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  std::vector<NamedAttribute> attrs;
  auto weight_bits_attr =
      rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
  attrs.push_back(weight_bits_attr);

  auto w_transpose_attr =
      rewriter.getNamedAttr("w_transpose", op.getRightTransposeAttr());
  attrs.push_back(w_transpose_attr);

  auto q_group_size_attr =
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr());
  attrs.push_back(q_group_size_attr);

  rewriter.replaceOpWithNewOp<tpu::A16MatMulOp>(op, newType, operands, attrs);
}

void A16MatMulLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::A16MatMulOp op) const {
  llvm_unreachable("Not implement");
}

void A16MatMulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::A16MatMulOp op) const {
  llvm_unreachable("Not implement");
}

} // namespace bm1684x
} // namespace tpu_mlir
