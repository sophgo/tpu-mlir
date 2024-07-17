//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"

namespace tpu_mlir {
namespace bm1684 {

void MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_f32<tpu::MatMulOp>(rewriter, op, 5);
}

void MatMulLowering::LoweringINT8(PatternRewriter &rewriter, top::MatMulOp op,
                                  bool asymmetric) const {
  // refer quantize_convlike_layer_int8
  OpBuilder builder(op);
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto p = op.parseParam();
  auto in_shape = module::getShape(op.getInput());
  auto r_shape = module::getShape(op.getRight());
  if (module::isWeight(op.getRight()) && in_shape.size() > 0 &&
      r_shape.size() > 0 && in_shape[0] == p.M && r_shape[0] == p.K) {
    assert(p.batch == 1); // only for fullyconnected now
    int64_t in_zp = 0, out_zp = 0;
    double in_scale = 1, out_scale = 1;
    module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
    module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
    auto filterOp = cast<top::WeightOp>(op.getRight().getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    double filter_max = findMaxabs(filter_f32->data(), filter_f32->size());
    int rshift = calRightShiftNum(filter_max, in_scale, out_scale, BITS_INT8);
    rshift = rshift >= 0 ? rshift : 0;
    std::shared_ptr<std::vector<int16_t>> bias_int16;
    if (p.with_bias) {
      float bias_scale = 1.0 * (1 << rshift) / out_scale;
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto bias_f32 = biasOp.read<float>();
      bias_int16 = std::make_shared<std::vector<int16_t>>(bias_f32->size());
      float overflow_ratio = quantizeToInt16(
          bias_f32->data(), bias_int16->data(), bias_f32->size(), bias_scale);

      while (overflow_ratio > 0.03 && rshift > 0) {
        rshift--;
        bias_scale = 1.0 * (1 << rshift) / out_scale;
        overflow_ratio = quantizeToInt16(bias_f32->data(), bias_int16->data(),
                                         bias_f32->size(), bias_scale);
      }
    }
    attrs.push_back(
        rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift)));
    float scale = 1.0 * (1 << rshift) * in_scale / out_scale;
    auto filter_int8 =
        std::make_shared<std::vector<int8_t>>(filter_f32->size());
    quantizeToInt8(filter_f32->data(), filter_int8->data(), filter_f32->size(),
                   scale);
    auto filter_type = op.getRight().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
    auto new_filter =
        top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
    operands.push_back(op.getInput());
    operands.push_back(new_filter);
    auto new_bias = op.getBias();
    if (p.with_bias) {
      auto bias_type = op.getBias().getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(bias_type.getShape(),
                                            rewriter.getIntegerType(16));
      new_bias = top::WeightOp::create(op, "bias_int16", *bias_int16, new_type);
    }
    operands.push_back(new_bias);
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    auto newType = getQuantInt8Type(op.getOutput());
    auto noneOp_multi = module::getNoneOp(op);
    operands.push_back(noneOp_multi);
    // buffer
    operands.push_back(module::getNoneOp(op));
    rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
  } else {
    lowering_common_f32<tpu::MatMulOp>(rewriter, op, 5);
  }
}

} // namespace bm1684
} // namespace tpu_mlir
