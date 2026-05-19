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

void Fp8MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::Fp8MatMulOp op) const {
  llvm_unreachable("Not implement");
}

void Fp8MatMulLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::Fp8MatMulOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void Fp8MatMulLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::Fp8MatMulOp op,
                                     bool asymmetric) const {
  llvm_unreachable("Not implement");
}

void Fp8MatMulLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::Fp8MatMulOp op) const {
  auto newType = getQuantBF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));

  auto weight_value = op.getWeight();
  operands.push_back(weight_value);

  // lowering scales
  auto scale_value = op.getWeightScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  auto new_scale_value = scale_op.clone_bf16(op);
  operands.push_back(new_scale_value);

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

  auto w_transpose_attr =
      rewriter.getNamedAttr("weight_transpose", op.getWeightTransposeAttr());
  attrs.push_back(w_transpose_attr);

  auto block_size_attr =
      rewriter.getNamedAttr("block_size", op.getBlockSizeAttr());
  attrs.push_back(block_size_attr);

  rewriter.replaceOpWithNewOp<tpu::Fp8MatMulOp>(op, newType, operands, attrs);
}

void Fp8MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::Fp8MatMulOp op) const {
  auto newType = getQuantF16Type(op->getResult(0));
  std::vector<Value> operands;

  // add input
  operands.push_back(op->getOperand(0));

  auto weight_value = op.getWeight();
  operands.push_back(weight_value);

  // lowering scales
  auto scale_value = op.getWeightScale();
  auto scale_op = dyn_cast<top::WeightOp>(scale_value.getDefiningOp());
  auto new_scale_value = scale_op.clone_f16(op);
  operands.push_back(new_scale_value);

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

  auto w_transpose_attr =
      rewriter.getNamedAttr("weight_transpose", op.getWeightTransposeAttr());
  attrs.push_back(w_transpose_attr);

  auto block_size_attr =
      rewriter.getNamedAttr("block_size", op.getBlockSizeAttr());
  attrs.push_back(block_size_attr);

  rewriter.replaceOpWithNewOp<tpu::Fp8MatMulOp>(op, newType, operands, attrs);
}

void Fp8MatMulLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::Fp8MatMulOp op) const {
  llvm_unreachable("Not implement");
}

void Fp8MatMulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::Fp8MatMulOp op) const {
  llvm_unreachable("Not implement");
}

} // namespace bm1684x
} // namespace tpu_mlir
