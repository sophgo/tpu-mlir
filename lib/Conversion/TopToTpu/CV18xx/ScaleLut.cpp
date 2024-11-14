//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-ScaleLut"
namespace tpu_mlir {
namespace cv18xx {
void ScaleLutLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::ScaleLutOp op, bool asymmetric) const {
  // lowering_common_int8<tpu::ScaleLutOp>(rewriter, op, asymmetric);
  Value input_val = op.getInput();
  int64_t n, c, h, w;
  module::getNCHW(input_val, n, c, h, w);
  std::string name = module::getName(op.getOutput()).str();
  auto scale = module::getF64Array(op.getScale());
  auto bias = module::getF64Array(op.getBias());
  int table_h = 16;
  int table_w = 16;
  int table_hw = table_h * table_w;
  int table_size = c * table_hw;
  std::vector<int8_t> table(table_size, 0);
  for (int i = 0; i < c; i++) {
    for (int idx = 0; idx < table_hw; ++idx) {
      table[i * table_hw + idx] =
          to_int8(idx * scale->at(i) + bias->at(i), ROUNDING_HALF_UP);
    }
  }
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
  attrs.emplace_back(rewriter.getNamedAttr("bias", op.getBiasAttr()));
  auto table_shape = std::vector<int64_t>{1, c, table_h, table_w};
  auto table_type = RankedTensorType::get(table_shape, rewriter.getI8Type());
  auto table_op = top::WeightOp::create(op, name + "_table", table, table_type);
  std::vector<Value> operands;
  operands.emplace_back(input_val);
  operands.emplace_back(table_op);
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ScaleLutOp>(op, newType, operands, attrs);
}

void ScaleLutLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::ScaleLutOp op) const {
  LoweringINT8(rewriter, op, false);
}
} // namespace cv18xx
} // namespace tpu_mlir
