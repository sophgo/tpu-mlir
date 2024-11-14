//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

#define DEBUG_TYPE "lowering-ScaleLut"
namespace tpu_mlir {
namespace bm1684x {
void ScaleLutLowering::LoweringINT8(PatternRewriter &rewriter,
                                    top::ScaleLutOp op, bool asymmetric) const {
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
  bool sign = op.getSign();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
  attrs.emplace_back(rewriter.getNamedAttr("bias", op.getBiasAttr()));
  attrs.emplace_back(rewriter.getNamedAttr("sign", op.getSignAttr()));
  auto table_shape = std::vector<int64_t>{1, c, table_h, table_w};
  std::vector<Value> operands;
  operands.emplace_back(input_val);

  if (!sign) {
    std::vector<uint8_t> table(table_size, 0);
    auto table_type =
        RankedTensorType::get(table_shape, rewriter.getIntegerType(8, false));
    for (int i = 0; i < c; i++) {
      for (int idx = 0; idx < table_hw; ++idx) {
        table[i * table_hw + idx] =
            to_uint8(idx * scale->at(i), ROUNDING_HALF_UP);
      }
    }
    auto table_op =
        top::WeightOp::create(op, name + "_table", table, table_type);
    operands.emplace_back(table_op);

    auto output = op.getOutput();
    auto type = output.getType().cast<RankedTensorType>();
    auto ctx = output.getContext();
    auto cali_type = module::getCalibratedType(output);
    double scale;
    int64_t zeropoint = 0;
    int64_t qmin = 0, qmax = 255;
    uint32_t flag = 0;
    auto max = cali_type.getMax();
    scale = module::getScale(max, sign);
    auto qtype = quant::UniformQuantizedType::get(
        flag, IntegerType::get(ctx, 8), cali_type.getExpressedType(), scale,
        zeropoint, qmin, qmax);
    auto newType = RankedTensorType::get(type.getShape(), qtype);
    rewriter.replaceOpWithNewOp<tpu::ScaleLutOp>(op, newType, operands, attrs);
  } else {
    std::vector<int8_t> table(table_size, 0);
    auto table_type = RankedTensorType::get(table_shape, rewriter.getI8Type());
    for (int i = 0; i < c; i++) {
      for (int idx = 0; idx < table_hw; ++idx) {
        table[i * table_hw + idx] =
            to_int8(idx * scale->at(i) + bias->at(i), ROUNDING_HALF_UP);
      }
    }
    auto table_op =
        top::WeightOp::create(op, name + "_table", table, table_type);
    operands.emplace_back(table_op);

    auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
    rewriter.replaceOpWithNewOp<tpu::ScaleLutOp>(op, newType, operands, attrs);
  }
}

void ScaleLutLowering::LoweringF32(PatternRewriter &rewriter,
                                   top::ScaleLutOp op) const {
  lowering_common_f32<tpu::ScaleLutOp>(rewriter, op);
}

void ScaleLutLowering::LoweringINT4(PatternRewriter &rewriter,
                                    top::ScaleLutOp op, bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void ScaleLutLowering::LoweringF16(PatternRewriter &rewriter,
                                   top::ScaleLutOp op) const {
  LoweringF32(rewriter, op);
}

void ScaleLutLowering::LoweringBF16(PatternRewriter &rewriter,
                                    top::ScaleLutOp op) const {
  LoweringF32(rewriter, op);
}

void ScaleLutLowering::LoweringF8(PatternRewriter &rewriter,
                                  top::ScaleLutOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ScaleLutLowering::LoweringQuantized(PatternRewriter &rewriter,
                                         top::ScaleLutOp op) const {
  lowering_common<tpu::ScaleLutOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
