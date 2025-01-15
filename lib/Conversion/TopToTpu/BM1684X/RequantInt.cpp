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

void RequantIntLowering::LoweringF32(PatternRewriter &rewriter,
                                     top::RequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantIntLowering::LoweringINT8(PatternRewriter &rewriter,
                                      top::RequantIntOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RequantIntLowering::LoweringINT4(PatternRewriter &rewriter,
                                      top::RequantIntOp op,
                                      bool asymmetric) const {
  UNREACHABLE_OP("Not Implemented", op);
}
void RequantIntLowering::LoweringBF16(PatternRewriter &rewriter,
                                      top::RequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantIntLowering::LoweringF16(PatternRewriter &rewriter,
                                     top::RequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantIntLowering::LoweringF8(PatternRewriter &rewriter,
                                    top::RequantIntOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void RequantIntLowering::LoweringQuantized(PatternRewriter &rewriter,
                                           top::RequantIntOp op) const {

  auto o_qtype = module::getUniformQuantizedType(op.getOutput());
  auto shift = module::getI64Array(op.getRshift());
  auto multi = module::getI64Array(op.getMultiplier());
  auto requant_mode = op.getQuantModeAttr().str();
  auto round_mode = op.getRoundModeAttr().str();
  auto zero_point = o_qtype.getZeroPoint();
  int32_t rq_axis = op.getRqAxis();
  bool fuse_rq = op.getFuseRq();
  auto raw_shift = *shift;
  auto raw_multi = *multi;

  if (raw_multi.size() == 1) {
    auto newValue =
        do_requant(op->getLoc(), op.getInput(), op.getOutput().getType(), true,
                   raw_multi[0], -raw_shift[0], get_requant_mode(requant_mode),
                   get_round_mode(round_mode));
    rewriter.replaceOp(op, {newValue});
  } else {
    std::vector<int32_t> quant;
    std::vector<int64_t> in_shape = module::getShape(op.getInput());
    std::vector<int64_t> quant_shape(in_shape.size(), 1l);
    bool isBM1688 = module::isBM1688() || module::isSG2380() ||
                    module::isMARS3() || module::isSGTPUV8();
    int numElementsPerChannel = isBM1688 ? 2 : 3;
    quant.resize(raw_multi.size() * numElementsPerChannel, 0);
    for (int i = 0; i < raw_multi.size(); ++i) {
      if (isBM1688) {
        quant[i * 2] = raw_multi[i];
        quant[i * 2 + 1] = raw_shift.size() == 1
                               ? ((-(int32_t)raw_shift[0]) & 0xffff) |
                                     (((int32_t)zero_point & 0xffff) << 16)
                               : ((-(int32_t)raw_shift[i]) & 0xffff) |
                                     (((int32_t)zero_point & 0xffff) << 16);
      } else {
        quant[i * 3] = raw_multi[i];
        quant[i * 3 + 1] =
            raw_shift.size() == 1 ? -raw_shift[0] : -raw_shift[i];
        quant[i * 3 + 2] = zero_point;
      }
    }

    if (fuse_rq) {
      assert(quant_shape.size() - 2 >= 0);
      quant_shape[quant_shape.size() - 2] = numElementsPerChannel;
      quant_shape.back() = raw_multi.size();
    } else {
      quant_shape.back() = numElementsPerChannel;
      quant_shape[1] = raw_multi.size();
    }

    auto quant_type = RankedTensorType::get(quant_shape, rewriter.getI32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    auto original_name = module::getName(op.getOperation()).str();
    auto newValue = do_requant_axis(
        op.getLoc(), op.getInput(), quantValue, op.getOutput().getType(), true,
        get_requant_mode(requant_mode), get_round_mode(round_mode), rq_axis,
        fuse_rq);

    rewriter.replaceOp(op, {newValue});
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
