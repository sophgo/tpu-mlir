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

static void LoweringTile(PatternRewriter &rewriter, top::TileOp op, Type type) {
  rewriter.setInsertionPointAfter(op);
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
  auto new_type = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tpu::TileOp>(op, new_type, operands, attrs);
  return;
}

void TileTryLowering::Lowering(PatternRewriter &rewriter,
                                  top::TileOp op) const {
  if (!op.getTileT() ||
      (!op.getTileT().getDefiningOp()->hasTrait<trait::ShapeProducer>()))
    return;
  LoweringTile(rewriter, op, rewriter.getF32Type());
}

void TileLowering::LoweringF32(PatternRewriter &rewriter,
                               top::TileOp op) const {
  LoweringTile(rewriter, op, rewriter.getF32Type());

}

void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp op,
                                bool asymmetric) const {
  lowering_common_int8<tpu::TileOp>(rewriter, op, asymmetric, 2);
}
void TileLowering::LoweringINT4(PatternRewriter &rewriter, top::TileOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp op) const {
  lowering_common_bf16<tpu::TileOp>(rewriter, op, 2);
}

void TileLowering::LoweringF16(PatternRewriter &rewriter,
                               top::TileOp op) const {
  lowering_common_f16<tpu::TileOp>(rewriter, op, 2);
}

void TileLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::TileOp op) const {
  lowering_common<tpu::TileOp>(rewriter, op, op.getOutput().getType(), 2);
}

} // namespace bm1684x
} // namespace tpu_mlir
