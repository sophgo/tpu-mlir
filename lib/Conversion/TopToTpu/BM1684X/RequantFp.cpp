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

void RequantFpLowering::LoweringF32(PatternRewriter &rewriter,
                                    top::RequantFpOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantFpLowering::LoweringINT8(PatternRewriter &rewriter,
                                     top::RequantFpOp op,
                                     bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RequantFpLowering::LoweringINT4(PatternRewriter &rewriter,
                                     top::RequantFpOp op,
                                     bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RequantFpLowering::LoweringBF16(PatternRewriter &rewriter,
                                     top::RequantFpOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantFpLowering::LoweringF16(PatternRewriter &rewriter,
                                    top::RequantFpOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantFpLowering::LoweringF8(PatternRewriter &rewriter,
                                   top::RequantFpOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantFpLowering::LoweringQuantized(PatternRewriter &rewriter,
                                          top::RequantFpOp op) const {
  auto scale = module::getF64Array(op.getScale());
  auto offset = module::getF64Array(op.getOffset());
  auto requant_mode = op.getQuantModeAttr().str();
  auto round_mode = op.getRoundModeAttr().str();
  auto first_round_mode = op.getFirstRoundModeAttr().str();
  auto raw_scale = *scale;
  auto raw_offset = *offset;
  assert(raw_scale.size() == raw_offset.size() &&
         "multi & shift size missmatch");
  auto name = module::getName(op.getOperation()).str();

  if (raw_scale.size() == 1) {
    auto newValue = do_requantFp(
        op.getInput(), raw_scale[0], raw_offset[0], op.getOutput().getType(),
        name, get_requant_mode(requant_mode), get_round_mode(round_mode),
        get_round_mode(first_round_mode));
    rewriter.replaceOp(op, {newValue});
  } else {
    std::vector<int32_t> quant;
    std::vector<int64_t> quant_shape(module::getShape(op.getInput()).size(),
                                     1l);
    quant_shape[1] = raw_scale.size();
    quant.resize(raw_scale.size() * 2, 0);
    for (int i = 0; i < raw_scale.size(); ++i) {
      quant[i * 2] = raw_scale[i];
      quant[i * 2 + 1] = (int32_t)raw_offset[i];
    }
    quant_shape.back() = 2;

    auto quant_type = RankedTensorType::get(quant_shape, rewriter.getF32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    auto newValue = do_requantFp(
        op.getInput(), quantValue, op.getOutput().getType(), true, name,
        get_requant_mode(requant_mode), get_round_mode(round_mode),
        get_round_mode(first_round_mode));
    rewriter.replaceOp(op, {newValue});
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
