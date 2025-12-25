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

void MlpLowering::LoweringF32(PatternRewriter &rewriter, top::MlpOp op) const {
  llvm_unreachable("Not implement");
}

void MlpLowering::LoweringINT8(PatternRewriter &rewriter, top::MlpOp op,
                               bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void MlpLowering::LoweringINT4(PatternRewriter &rewriter, top::MlpOp op,
                               bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void MlpLowering::LoweringBF16(PatternRewriter &rewriter, top::MlpOp op) const {
  auto newType = getQuantBF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));
  bool quantized = op.getQuantized();

  // lowering gate_proj
  // lowering weights
  auto weight_value = op.getWeightGate();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_bf16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleGate();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_bf16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpGate().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  auto bias_value = op.getBiasGate();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_bf16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // lowering up_proj
  // lowering weights
  weight_value = op.getWeightUp();
  weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_bf16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleUp();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_bf16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpUp().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  bias_value = op.getBiasUp();
  bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_bf16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // lowering down_proj
  // lowering weights
  weight_value = op.getWeightDown();
  weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_bf16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleDown();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_bf16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpDown().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  bias_value = op.getBiasDown();
  bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_bf16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // buffer
  operands.push_back(module::getNoneOp(op));
  // attrs
  std::vector<NamedAttribute> attrs;
  auto quantized_attr =
      rewriter.getNamedAttr("quantized", op.getQuantizedAttr());
  attrs.push_back(quantized_attr);

  auto weight_bits_attr =
      rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
  attrs.push_back(weight_bits_attr);

  auto w_transpose_gate_attr =
      rewriter.getNamedAttr("w_transpose_gate", op.getRightTransposeGateAttr());
  attrs.push_back(w_transpose_gate_attr);

  auto w_transpose_up_attr =
      rewriter.getNamedAttr("w_transpose_up", op.getRightTransposeUpAttr());
  attrs.push_back(w_transpose_up_attr);

  auto w_transpose_down_attr =
      rewriter.getNamedAttr("w_transpose_down", op.getRightTransposeDownAttr());
  attrs.push_back(w_transpose_down_attr);

  auto q_group_size_attr =
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr());
  attrs.push_back(q_group_size_attr);

  rewriter.replaceOpWithNewOp<tpu::MlpOp>(op, newType, operands, attrs);
}

void MlpLowering::LoweringF16(PatternRewriter &rewriter, top::MlpOp op) const {
  auto newType = getQuantF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));
  bool quantized = op.getQuantized();

  // lowering gate_proj
  // lowering weights
  auto weight_value = op.getWeightGate();
  auto weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_f16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleGate();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_f16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpGate().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  auto bias_value = op.getBiasGate();
  auto bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_f16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // lowering up_proj
  // lowering weights
  weight_value = op.getWeightUp();
  weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_f16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleUp();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_f16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpUp().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  bias_value = op.getBiasUp();
  bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_f16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // lowering down_proj
  // lowering weights
  weight_value = op.getWeightDown();
  weight_op = dyn_cast<top::WeightOp>(weight_value.getDefiningOp());
  if (quantized) {
    auto weight_data = weight_op.read<int8_t>();
    auto new_weight_type = RankedTensorType::get(
        weight_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_weight_value =
        top::WeightOp::create(op, "weight", *weight_data, new_weight_type);
    operands.push_back(new_weight_value);
  } else {
    operands.push_back(weight_op.clone_f16(op));
  }
  // lowering scales
  if (quantized) {
    auto scale_value = op.getScaleDown();
    auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
    auto new_scale_value = scale_op.clone_f16(op);
    operands.push_back(new_scale_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering zps
  if (quantized) {
    auto zp_op = op.getZpDown().getDefiningOp<top::WeightOp>();
    auto zp_data = zp_op.read<int8_t>();
    auto new_zp_type = RankedTensorType::get(
        zp_op.getType().cast<RankedTensorType>().getShape(),
        rewriter.getIntegerType(8, false));
    auto new_zp_value = top::WeightOp::create(op, "zp", *zp_data, new_zp_type)
                            .getDefiningOp<top::WeightOp>();
    operands.push_back(new_zp_value);
  } else {
    operands.push_back(module::getNoneOp(op));
  }
  // lowering bias
  bias_value = op.getBiasDown();
  bias_op = dyn_cast<top::WeightOp>(bias_value.getDefiningOp());
  if (bias_op) {
    operands.push_back(bias_op.clone_f16(op));
  } else {
    auto noneOp = module::getNoneOp(op);
    operands.push_back(noneOp);
  }

  // buffer
  operands.push_back(module::getNoneOp(op));
  // attrs
  std::vector<NamedAttribute> attrs;
  auto quantized_attr =
      rewriter.getNamedAttr("quantized", op.getQuantizedAttr());
  attrs.push_back(quantized_attr);

  auto weight_bits_attr =
      rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
  attrs.push_back(weight_bits_attr);

  auto w_transpose_gate_attr =
      rewriter.getNamedAttr("w_transpose_gate", op.getRightTransposeGateAttr());
  attrs.push_back(w_transpose_gate_attr);

  auto w_transpose_up_attr =
      rewriter.getNamedAttr("w_transpose_up", op.getRightTransposeUpAttr());
  attrs.push_back(w_transpose_up_attr);

  auto w_transpose_down_attr =
      rewriter.getNamedAttr("w_transpose_down", op.getRightTransposeDownAttr());
  attrs.push_back(w_transpose_down_attr);

  auto q_group_size_attr =
      rewriter.getNamedAttr("q_group_size", op.getQGroupSizeAttr());
  attrs.push_back(q_group_size_attr);

  rewriter.replaceOpWithNewOp<tpu::MlpOp>(op, newType, operands, attrs);
}

void MlpLowering::LoweringF8(PatternRewriter &rewriter, top::MlpOp op) const {
  llvm_unreachable("Not implement");
}

void MlpLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::MlpOp op) const {
  llvm_unreachable("Not implement");
}

} // namespace bm1684x
} // namespace tpu_mlir
