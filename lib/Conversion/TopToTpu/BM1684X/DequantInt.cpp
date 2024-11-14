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

void DequantIntLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::DequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void DequantIntLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::DequantIntOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void DequantIntLowering::LoweringINT4(PatternRewriter &rewriter,
                                      top::DequantIntOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void DequantIntLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::DequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void DequantIntLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::DequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void DequantIntLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::DequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void DequantIntLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::DequantIntOp op) const {
  // lowering_common<tpu::DequantIntOp>(rewriter, op.getOperation(),
  //                             op.getOutput().getType());
  auto shift = module::getI64Array(op.getShift());
  auto multi = module::getI64Array(op.getMultiplier());
  auto dequant_mode = op.getQuantModeAttr().str();
  auto round_mode = op.getRoundModeAttr().str();
  auto raw_shift = *shift;
  auto raw_multi = *multi;
  assert(raw_multi.size() == raw_shift.size() &&
         "multi & shift size missmatch");

  if (raw_multi.size() == 1) {
    auto newValue =
        do_dequant(op->getLoc(), op.getInput(), op.getOutput().getType(),
                   raw_multi[0], raw_shift[0], get_dequant_mode(dequant_mode),
                   op.getLshift(), get_round_mode(round_mode));
    rewriter.replaceOp(op, {newValue});
  } else {
    UNREACHABLE_OP("Not Implemented", op);
    // std::vector<int32_t> quant;
    // std::vector<int64_t> quant_shape(module::getShape(op.getInput()).size(),
    //                                  1l);
    // quant_shape[1] = raw_multi.size();
    // if (module::isBM1688()) {
    //   quant.resize(raw_multi.size() * 2, 0);
    //   for (int i = 0; i < raw_multi.size(); ++i) {
    //     quant[i * 2] = raw_multi[i];
    //     quant[i * 2 + 1] =
    //         (((int32_t)raw_shift[i]) & 0xffff) |
    //         (((int32_t)zero_point & 0xffff) << 16);
    //   }
    //   quant_shape.back() = 2;
    // } else {
    //   quant.resize(raw_multi.size() * 3, 0);
    //   for (int i = 0; i < raw_multi.size(); ++i) {
    //     quant[i * 3] = raw_multi[i];
    //     quant[i * 3 + 1] = raw_shift[i];
    //     quant[i * 3 + 2] = zero_point;
    //   }
    //   quant_shape.back() = 3;
    // }
    // auto quant_type = RankedTensorType::get(quant_shape,
    // rewriter.getI32Type()); auto quantValue = top::WeightOp::create(op,
    // "quant", quant, quant_type); auto newValue =
    //     do_dequant(op->getLoc(), op.getInput(), quantValue,
    //     op.getOutput().getType(),
    //                true, tpu::DequantMode::TFLite_LShift);
    // rewriter.replaceOp(op, {newValue});
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
