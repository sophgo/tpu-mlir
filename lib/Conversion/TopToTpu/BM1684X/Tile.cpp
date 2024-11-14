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

static void _try_insert_device2host(top::TileOp op) {
  if (op.getTileT()) {
    try_insert_device2host(op.getOperation(), 1);
  }
}

static void LoweringTile(PatternRewriter &rewriter, top::TileOp op, Type type) {
  _try_insert_device2host(op);

  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (op.getTileT())
    operands.push_back(op.getTileT());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  rewriter.replaceOpWithNewOp<tpu::TileOp>(op, type, operands, attrs);
  return;
}

void TileLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TileOp op) const {
  auto new_type = getQuantFloatType(op->getResult(0));
  LoweringTile(rewriter, op, new_type);
}

void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp op,
                                bool asymmetric) const {
  if (op.getTileT()) {
    LoweringF32(rewriter, op);
  } else {
    lowering_common_int8<tpu::TileOp>(rewriter, op, asymmetric, 2);
  }
}
void TileLowering::LoweringINT4(PatternRewriter &rewriter, top::TileOp op,
                                bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp op) const {
  if (op.getTileT()) {
    LoweringF32(rewriter, op);
  } else {
    lowering_common_bf16<tpu::TileOp>(rewriter, op, 2);
  }
}

void TileLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TileOp op) const {
  if (op.getTileT()) {
    LoweringF32(rewriter, op);
  } else {
    lowering_common_f16<tpu::TileOp>(rewriter, op, 2);
  }
}

void TileLowering::LoweringF8(PatternRewriter &rewriter, top::TileOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void TileLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TileOp op) const {
  if (op.getTileT()) {
    LoweringF32(rewriter, op);
  } else {
    lowering_common<tpu::TileOp>(rewriter, op, op.getOutput().getType(), 2);
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
