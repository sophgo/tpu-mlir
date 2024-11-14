//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

namespace tpu_mlir {
namespace cv18xx {

static void LoweringTile(PatternRewriter &rewriter, top::TileOp &op,
                         bool quant_i8) {
  if (op.getTileT()) {
    llvm_unreachable("Not support dynamic Tile.");
  }

  auto newOp = op.getInput();
  Type eltType;
  if (quant_i8) {
    eltType = getQuantInt8Type(op.getInput());
  } else {
    eltType = getQuantBF16Type(op.getInput());
  }
  // Type eltType = getQuantBF16Type(op.getInput());
  auto output_shape = module::getShape(op.getOutput());
  auto input_shape = module::getShape(op.getInput());
  auto output_dims = output_shape.size();
  auto op_name = module::getName(op.getOperation()).str();
  int count = 0;
  std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
  for (uint32_t i = 0; i < output_dims; i++) {
    if (out_shape[i] != output_shape[i]) {
      ++count;
    }
  }
  std::vector<int64_t> weight_tile(output_dims, 1);
  std::vector<NamedAttribute> attrs;
  std::vector<Value> operands;
  operands.push_back(newOp);
  auto noneOp = module::getNoneOp(op);
  operands.push_back(noneOp);
  for (uint32_t i = 1; i <= output_dims; i++) {
    if (input_shape[output_dims - i] == output_shape[output_dims - i])
      continue;
    weight_tile[output_dims - i] =
        output_shape[output_dims - i] / out_shape[output_dims - i];
    out_shape[output_dims - i] = output_shape[output_dims - i];
    attrs.push_back(
        rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr(weight_tile)));
    auto result_type = RankedTensorType::get(
        out_shape, eltType.cast<RankedTensorType>().getElementType());
    NameLoc loc;
    if (count == 1) {
      loc = NameLoc::get(rewriter.getStringAttr(op_name));
    } else {
      loc =
          NameLoc::get(rewriter.getStringAttr(op_name + std::to_string(count)));
    }
    auto newTile =
        rewriter.create<tpu::TileOp>(loc, result_type, operands, attrs);
    weight_tile[output_dims - i] = 1;
    attrs.clear();
    count--;
    operands.clear();
    operands.push_back(newTile.getOutput());
    operands.push_back(noneOp);
    if (count == 0) {
      rewriter.replaceAllUsesWith(op.getOutput(), newTile.getOutput());
      rewriter.eraseOp(op);
      return;
    }
  }
}

// todo num_dims > 4, do loop tile
void TileLowering::LoweringINT8(PatternRewriter &rewriter, top::TileOp op,
                                bool asymmetric) const {
  return LoweringTile(rewriter, op, true);
}

void TileLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::TileOp op) const {
  return LoweringTile(rewriter, op, false);
}

} // namespace cv18xx
} // namespace tpu_mlir
