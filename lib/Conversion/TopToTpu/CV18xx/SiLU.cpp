//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-SiLU"

namespace tpu_mlir {
namespace cv18xx {

static double active_silu(double val) { return val / (1 + std::exp(-val)); }

void SiLULowering::LoweringINT8(PatternRewriter &rewriter, top::SiLUOp op,
                                bool asymmetric) const {
  OpBuilder builder(op->getContext());
  auto stype = module::getStorageType(op.getOutput());
  auto table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric, [](double val) {
        return val / (1 + std::exp(-val));
      });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
  return;
}

void SiLULowering::LoweringBF16(PatternRewriter &rewriter,
                                top::SiLUOp op) const {
  Value table_weight, slope_weight;
  float range_start = -12, range_end = 12;
  createBf16LutOp(op, "slope", TableMode::Slope, 0.0, 0.0, range_start,
                  range_end, active_silu, table_weight, slope_weight);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.emplace_back(attr);
  }
  attrs.push_back(rewriter.getNamedAttr(
      "lut_mode",
      tpu::LutBF16ModeAttr::get(op->getContext(), tpu::LutBF16Mode::Slope)));
  attrs.push_back(rewriter.getNamedAttr("min_range",
                                        rewriter.getF64FloatAttr(range_start)));
  attrs.push_back(
      rewriter.getNamedAttr("max_range", rewriter.getF64FloatAttr(range_end)));
  auto newType = getQuantBF16Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::LutBF16Op>(
      op, newType, ValueRange{op.getInput(), table_weight, slope_weight}, attrs);
  return;
}

} // namespace cv18xx
} // namespace tpu_mlir
