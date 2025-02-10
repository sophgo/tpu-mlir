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
static void LoweringConvbwd(PatternRewriter &rewriter, top::ConvbwdOp op,
                            Type type) {
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (auto &&in : op.getOperands())
    operands.emplace_back(in);
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  std::vector<Type> new_types;
  new_types.reserve(3);
  for (int i = 0; i < 3; i++) {
    auto out = op.getResult(i);
    if (type.isF16()) {
      new_types.push_back(getQuantF16Type(out));
    } else if (type.isBF16()) {
      new_types.push_back(getQuantBF16Type(out));
    } else {
      new_types.push_back(out.getType());
    }
  }
  rewriter.replaceOpWithNewOp<tpu::ConvbwdOp>(op, new_types, operands,
                                              op.getOperation()->getAttrs());
}

void ConvbwdLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::ConvbwdOp op) const {
  // lowering_common_f32<tpu::ConvbwdOp>(rewriter, op);
  LoweringConvbwd(rewriter, op, rewriter.getF32Type());
}
void ConvbwdLowering::LoweringINT4(PatternRewriter &rewriter, top::ConvbwdOp op,
                                   bool asymmetric) const {
  // LoweringINT8(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}
void ConvbwdLowering::LoweringINT8(PatternRewriter &rewriter, top::ConvbwdOp op,
                                   bool asymmetric) const {
  // lowering_common_int8<tpu::ConvbwdOp>(rewriter, op, asymmetric);
  UNREACHABLE_OP("Not Implemented", op);
}

void ConvbwdLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::ConvbwdOp op) const {
  LoweringF32(rewriter, op);
}

void ConvbwdLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::ConvbwdOp op) const {
  // lowering_common_f16<tpu::ConvbwdOp>(rewriter, op);
  std::vector<Value> operands;
  std::vector<Type> new_types;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  for (auto in : op.getOperands()) {
    operands.push_back(in);
  }
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);
  auto NoneOp_0 = builder.create<top::NoneOp>(builder.getUnknownLoc(),
                                              builder.getNoneType());
  operands.push_back(NoneOp_0);
  auto grad_input_enable = op->getAttr("grad_input_enable").cast<BoolAttr>();
  if (grad_input_enable.getValue()) {
    new_types.push_back(getQuantF16Type(op.getResult(0)));
  } else {
    new_types.push_back(builder.getNoneType());
  }
  auto grad_weight_enable = op->getAttr("grad_weight_enable").cast<BoolAttr>();
  if (grad_weight_enable.getValue()) {
    new_types.push_back(getQuantF16Type(op.getResult(1)));
  } else {
    new_types.push_back(builder.getNoneType());
  }

  auto grad_bias_enable = op->getAttr("grad_bias_enable").cast<BoolAttr>();
  if (grad_bias_enable.getValue()) {
    new_types.push_back(getQuantF16Type(op.getResult(2)));
  } else {
    new_types.push_back(builder.getNoneType());
  }
  rewriter.replaceOpWithNewOp<tpu::ConvbwdOp>(op, new_types, operands, attrs);
}

void ConvbwdLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::ConvbwdOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void ConvbwdLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::ConvbwdOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
