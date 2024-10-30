//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

namespace tpu_mlir {
namespace bm1684x {

void ReverseTryLowering::Lowering(PatternRewriter &rewriter,
                                  top::ReverseOp op) const {
  if (!isa_shape_subnet_op(op))
    return;

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  auto v = op.getResult();
  auto shape = module::getShape(v);
  auto ctx = v.getContext();
  Type new_type = RankedTensorType::get(shape, IntegerType::get(ctx, 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeReverseOp>(op, new_type, op.getOperand(), attrs);
}

//ReverseOp to GatherOp in Mars3
void LoweringReverse(PatternRewriter &rewriter, top::ReverseOp op, Type type) {
  auto in_shape = module::getShape(op.getInput());
  int gather_dim = op.getAxis();
  int dim_length = in_shape[gather_dim];
  std::vector<int32_t> indices;
  for (int i = 0; i < dim_length; i++) {
    indices.push_back(dim_length - i - 1);
  };
  auto coeff_type = RankedTensorType::get(dim_length, rewriter.getIntegerType(32, true));
  auto indices_op = top::WeightOp::create(op, "indices", indices, coeff_type);

  std::vector<Value> operands;
  if (!module::isWeight(op.getInput())){
    operands.push_back(op.getInput());
  }
  operands.push_back(indices_op);
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);

  std::vector<NamedAttribute> attrs;
  bool keepdims = true;
  attrs.push_back(rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(op.getAxis())));
  attrs.push_back(rewriter.getNamedAttr("keepdims", rewriter.getBoolAttr(keepdims)));

  rewriter.replaceOpWithNewOp<tpu::GatherOp>(op, type, operands, attrs);
  return;
}

void ReverseLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  lowering_common_f32<tpu::ReverseOp>(rewriter, op);
}

void ReverseLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::ReverseOp op,
                                     bool asymmetric) const {
  if(module::isMARS3()){
    auto new_type = getQuantInt8Type(op.getOutput());
    LoweringReverse(rewriter, op, new_type);
  } else
    LoweringF32(rewriter, op);
}
void ReverseLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::ReverseOp op,
                                     bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void ReverseLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::ReverseOp op) const {
  if(module::isMARS3()){
    auto new_type = getQuantFloatType<mlir::BFloat16Type>(op.getOutput());
    LoweringReverse(rewriter, op, new_type);
  } else
    LoweringF32(rewriter, op);
}

void ReverseLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  LoweringF32(rewriter, op);
}

void ReverseLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::ReverseOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ReverseLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::ReverseOp op) const {
  LoweringF32(rewriter, op);
}

} // namespace bm1684x
} // namespace tpu_mlir
