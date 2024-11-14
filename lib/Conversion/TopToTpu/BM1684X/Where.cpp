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

void WhereLowering::LoweringF32(PatternRewriter &rewriter,
                                top::WhereOp op) const {
  lowering_common_f32<tpu::WhereOp>(rewriter, op, 4);
}
void WhereLowering::LoweringINT4(PatternRewriter &rewriter, top::WhereOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void WhereLowering::LoweringINT8(PatternRewriter &rewriter, top::WhereOp op,
                                 bool asymmetric) const {
  if (auto weight = dyn_cast<top::WeightOp>(op.getCond().getDefiningOp())) {
    float min, max;
    auto weight_f32 = weight.read<float>();
    findMinMax(weight_f32->data(), weight_f32->size(), &min, &max);
    auto weight_int8 =
        std::make_shared<std::vector<uint8_t>>(weight_f32->size());
    std::transform(weight_f32->begin(), weight_f32->end(), weight_int8->begin(),
                   [&](const float cf32) { return to_uint8(cf32); });
    auto weight_type = op.getCond().getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(weight_type.getShape(),
                                          rewriter.getIntegerType(8, false));
    auto new_weight = top::WeightOp::create(op, "i8", *weight_int8, new_type);
    op.setOperand(0, new_weight);
  }

  double scale;
  int64_t zp;
  module::getScaleAndZeroPoint(op.getOutput(), scale, zp, asymmetric, 8);
  double min = module::getCalibratedType(op.getOutput()).getMin();
  if (op.getXIsConst()) {
    double x_const_val = op.getXConstVal().convertToDouble();
    double new_x_const_val = x_const_val / scale + zp;
    new_x_const_val =
        min < 0 ? to_int8(new_x_const_val) : to_uint8(new_x_const_val);
    op.setXConstVal(APFloat(new_x_const_val));
  }
  if (op.getYIsConst()) {
    double y_const_val = op.getYConstVal().convertToDouble();
    double new_y_const_val = y_const_val / scale + zp;
    new_y_const_val =
        min < 0 ? to_int8(new_y_const_val) : to_uint8(new_y_const_val);
    op.setYConstVal(APFloat(new_y_const_val));
  }

  lowering_common_int8<tpu::WhereOp>(rewriter, op, asymmetric, 4);
}

void WhereLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::WhereOp op) const {
  lowering_common_bf16<tpu::WhereOp>(rewriter, op, 4);
}

void WhereLowering::LoweringF16(PatternRewriter &rewriter,
                                top::WhereOp op) const {
  lowering_common_f16<tpu::WhereOp>(rewriter, op, 4);
}

void WhereLowering::LoweringF8(PatternRewriter &rewriter,
                               top::WhereOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void WhereLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::WhereOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

} // namespace bm1684x
} // namespace tpu_mlir
