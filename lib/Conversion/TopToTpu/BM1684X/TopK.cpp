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

static void _try_insert_device2host(top::TopKOp op) {
  if (op.getKT()) {
    try_insert_device2host(op.getOperation(), 1);
  }
}

static void LoweringTopK(PatternRewriter &rewriter, top::TopKOp op, Type type) {
  if (module::isSG2380()) {
    std::vector<NamedAttribute> attrs;
    std::vector<NamedAttribute> cpu_param;
    attrs.emplace_back(
        rewriter.getNamedAttr("cpu_op_name", rewriter.getStringAttr("topk")));

    for (auto &attr : op->getAttrs()) {
      cpu_param.push_back(attr);
    }
    // If only values or indices are used, record it to support only one output
    // in cpu layer
    if (!op.getValues().getUsers().empty() &&
        op.getIndices().getUsers().empty()) {
      cpu_param.push_back(rewriter.getNamedAttr("values_used_only",
                                                rewriter.getBoolAttr(true)));
    } else {
      cpu_param.push_back(rewriter.getNamedAttr("values_used_only",
                                                rewriter.getBoolAttr(false)));
    }
    attrs.emplace_back(
        rewriter.getNamedAttr("param", rewriter.getDictionaryAttr(cpu_param)));
    std::vector<Type> new_types;
    new_types.push_back(op.getValues().getType());
    if (!module::isNone(op.getIndices())) {
      auto shape = module::getShape(op.getIndices());
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      new_types.push_back(new_type);
    } else {
      new_types.push_back(op.getIndices().getType());
    }
    std::vector<Value> operands;
    operands.push_back(op.getInput());
    rewriter.replaceOpWithNewOp<tpu::GenericCpuOp>(op, new_types, operands,
                                                   attrs);
    return;
  }

  _try_insert_device2host(op);

  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (op.getKT())
    operands.push_back(op.getKT());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  std::vector<Type> new_types;
  new_types.push_back(op.getValues().getType());
  if (!module::isNone(op.getIndices())) {
    auto shape = module::getShape(op.getIndices());
    auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
    new_types.push_back(new_type);
  } else {
    new_types.push_back(op.getIndices().getType());
  }

  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPoint(op);
  auto NoneOp_0 = builder.create<top::NoneOp>(builder.getUnknownLoc(),
                                              builder.getNoneType());
  auto NoneOp_1 = builder.create<top::NoneOp>(builder.getUnknownLoc(),
                                              builder.getNoneType());
  operands.push_back(NoneOp_0);
  operands.push_back(NoneOp_1);
  rewriter.replaceOpWithNewOp<tpu::TopKOp>(op, new_types, operands, attrs);
  return;
}

void TopKLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  LoweringTopK(rewriter, op, rewriter.getF32Type());
}
void TopKLowering::LoweringINT4(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}
void TopKLowering::LoweringINT8(PatternRewriter &rewriter, top::TopKOp op,
                                bool asymmetric) const {
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getBF16Type());
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getF16Type());
  LoweringF32(rewriter, op);
}

void TopKLowering::LoweringF8(PatternRewriter &rewriter, top::TopKOp op) const {
  // LoweringTopK(rewriter, op, rewriter.getF16Type());
  llvm_unreachable("FIXME: not implement");
}

void TopKLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TopKOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
