//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#define FP16_MAX 65504.0
#define FP16_MIN -65504.0
#define BF16_MAX 3.3895314e38
#define BF16_MIN -3.3895314e38
#define FP8E4M3_MAX 448.0
#define FP8E4M3_MIN -448.0
#define FP8E5M2_MAX 57344.0
#define FP8E5M2_MIN -57344.0

namespace tpu_mlir {
namespace bm1684x {

void MulConstLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  lowering_common_f32<tpu::MulConstOp>(rewriter, op);
}
void MulConstLowering::LoweringINT4(PatternRewriter &rewriter, top::MulConstOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void MulConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::MulConstOp op, bool asymmetric) const {
  double scale_i, scale_o;
  int64_t zp_i, zp_o;
  module::getScaleAndZeroPoint(op.getInput(), scale_i, zp_i, asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), scale_o, zp_o, asymmetric);
  auto scale = scale_i / scale_o * op.getConstVal().convertToDouble();
  int multiplier, rshift;
  get_scale_and_shift(scale, multiplier, rshift, 8);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType,
                                               ValueRange{op.getInput()}, attrs);
}

void MulConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::MulConstOp op) const {
  auto const_v = op.getConstVal().convertToDouble();
  if (const_v > BF16_MAX || const_v < BF16_MIN)
    LoweringF32(rewriter, op);
  else
    lowering_common_bf16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  auto const_v = op.getConstVal().convertToDouble();
  if (const_v > FP16_MAX || const_v < FP16_MIN)
    LoweringF32(rewriter, op);
  else
    lowering_common_f16<tpu::MulConstOp>(rewriter, op);
}

void MulConstLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::MulConstOp op) const {
  // llvm_unreachable("Not Implemented");
  auto const_v = op.getConstVal().convertToDouble();
  if ((const_v > FP8E5M2_MAX || const_v < FP8E5M2_MIN) && (module::getMode() == module::Mode::F8E5M2))
    LoweringF32(rewriter, op);
  else if ((const_v > FP8E4M3_MAX || const_v < FP8E4M3_MIN) && (module::getMode() == module::Mode::F8E4M3))
    LoweringF32(rewriter, op);
  else {
    double scale_i, scale_o;
    auto out = op.getOutput();
    auto qtype_out = module::getCalibratedType(out);

    if (module::getMode() == module::Mode::F8E5M2) {
      lowering_common_f8<tpu::MulConstOp>(rewriter, op, false);
      return ;
    }

    scale_o = qtype_out.getMax() / get_f8e4m3_max();
    auto input = op.getInput();
    auto qtype_in = module::getCalibratedType(input);
    scale_i = qtype_in.getMax() / get_f8e4m3_max();
    auto scale = scale_i / scale_o * op.getConstVal().convertToDouble();

    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }

    if (module::getMode() == module::Mode::F8E4M3) {
      attrs.push_back(
          rewriter.getNamedAttr("out_f8_scales", rewriter.getF64ArrayAttr(scale)));
      auto newType = getQuantF8E4M3Type(op.getOutput());
      rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType,
                                                 ValueRange{op.getInput()}, attrs);
    }
  }
}
void MulConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::MulConstOp op) const {
  llvm_unreachable("Not Implemented");
}

} // namespace bm1684x
} // namespace tpu_mlir
