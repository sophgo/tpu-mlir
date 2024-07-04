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

void Yuv2rgbFormulaLowering::LoweringF32(PatternRewriter &rewriter,
                                         top::Yuv2rgbFormulaOp op) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}

void Yuv2rgbFormulaLowering::LoweringINT8(PatternRewriter &rewriter,
                                          top::Yuv2rgbFormulaOp op,
                                          bool asymmetric) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}
void Yuv2rgbFormulaLowering::LoweringINT4(PatternRewriter &rewriter,
                                          top::Yuv2rgbFormulaOp op,
                                          bool asymmetric) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}
void Yuv2rgbFormulaLowering::LoweringBF16(PatternRewriter &rewriter,
                                          top::Yuv2rgbFormulaOp op) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}

void Yuv2rgbFormulaLowering::LoweringF16(PatternRewriter &rewriter,
                                         top::Yuv2rgbFormulaOp op) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}

void Yuv2rgbFormulaLowering::LoweringF8(PatternRewriter &rewriter,
                                        top::Yuv2rgbFormulaOp op) const {
  Yuv2rgbFormulaLowering::LoweringQuantized(rewriter, op);
}

void Yuv2rgbFormulaLowering::LoweringQuantized(
    PatternRewriter &rewriter, top::Yuv2rgbFormulaOp Yuv2rgbFormulaOp) const {
  //   auto op = Yuv2rgbFormulaOp.getOperation();
  auto outFormat = Yuv2rgbFormulaOp.getImageFormat();
  auto formulaMode = Yuv2rgbFormulaOp.getFormulaMode();
  auto roundMode = Yuv2rgbFormulaOp.getRoundMode();
  int outFormatNum = -1;
  int formulaModeNum = -1;
  int roundModeNum = -1;
  if (outFormat == "FLOAT32") {
    outFormatNum = 0;
  } else if (outFormat == "UINT8") {
    outFormatNum = 1;
  }

  if (formulaMode == "_601_limited") {
    formulaModeNum = 0;
  } else if (formulaMode == "_601_full") {
    formulaModeNum = 1;
  }

  if (roundMode == "HalfAwayFromZero") {
    roundModeNum = 0;
  } else if (roundMode == "HalfUp") {
    roundModeNum = 1;
  } else if (roundMode == "HalfToEven") {
    roundModeNum = 3;
  } else {
    llvm_unreachable("Not Implemented!");
  }

  Yuv2rgbFormulaOp->setAttr(
      "image_format",
      tpu::ImageOutFormatAttr::get(
          Yuv2rgbFormulaOp->getContext(),
          static_cast<tpu::ImageOutFormat>(tpu::ImageOutFormat(outFormatNum))));

  Yuv2rgbFormulaOp->setAttr(
      "formula_mode",
      tpu::Yuv2rgbFormulaAttr::get(Yuv2rgbFormulaOp->getContext(),
                                   static_cast<tpu::Yuv2rgbFormula>(
                                       tpu::Yuv2rgbFormula(formulaModeNum))));

  Yuv2rgbFormulaOp->setAttr(
      "round_mode", tpu::RoundModeAttr::get(Yuv2rgbFormulaOp->getContext(),
                                            static_cast<tpu::RoundMode>(
                                                tpu::RoundMode(roundModeNum))));

  lowering_common<tpu::Yuv2rgbFormulaOp>(
      rewriter, Yuv2rgbFormulaOp, Yuv2rgbFormulaOp->getResult(0).getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
