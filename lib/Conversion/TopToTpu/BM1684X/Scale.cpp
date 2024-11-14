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

void ScaleLowering::LoweringF32(PatternRewriter &rewriter,
                                top::ScaleOp op) const {
  auto ctx = op->getContext();
  auto output = op->getResult(0);
  auto sType = module::getStorageType(output);
  auto shape = module::getShape(output);
  Type newType = output.getType();
  if (sType.isa<FloatType>() == false) {
    if (module::isCalibratedType(output)) {
      auto caliType = module::getCalibratedType(output);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          Float32Type::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, Float32Type::get(ctx));
    }
  }

  OpBuilder builder(ctx);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  // lshift
  auto none = module::getNoneOp(op);
  operands.push_back(none);
  rewriter.replaceOpWithNewOp<tpu::ScaleOp>(op, newType, operands,
                                            op->getAttrs());
}
void ScaleLowering::LoweringINT4(PatternRewriter &rewriter, top::ScaleOp op,
                                 bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void ScaleLowering::LoweringINT8(PatternRewriter &rewriter, top::ScaleOp op,
                                 bool asymmetric) const {
  int64_t n, c, h, w;
  module::getNCHW(op.getOutput(), n, c, h, w);

  std::vector<Value> operands;
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);

  auto scaleOp = cast<top::WeightOp>(op.getScale().getDefiningOp());
  auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
  auto scale_f32 = scaleOp.read<float>();
  auto bias_f32 = biasOp.read<float>();
  auto scale_int8 = std::make_shared<std::vector<int8_t>>(scale_f32->size());
  auto lshift_int8 = std::make_shared<std::vector<int8_t>>(scale_f32->size());
  auto bias_int16 = std::make_shared<std::vector<int16_t>>(bias_f32->size());
  float scale_val;
  float overflow_ratio;
  for (int cidx = 0; cidx < c; cidx++) {
    // get right shift
    int rightShiftTmp = calRightShiftNumUseCblas(scale_f32->data()[cidx],
                                                 in_scale, out_scale, 8);
    rightShiftTmp++;
    do {
      rightShiftTmp--;
      scale_val = std::pow(2, rightShiftTmp) / out_scale;
      int16_t bias_int16_val;
      overflow_ratio = quantizeToInt16(bias_f32->data() + cidx, &bias_int16_val,
                                       1, scale_val);
      if (asymmetric) {
        scale_val = std::pow(2, rightShiftTmp) * in_scale / out_scale;
        int8_t tmp_scale = 1;
        quantizeToInt8(scale_f32->data() + cidx, &tmp_scale, 1, scale_val);
        int tmp_zp = tmp_scale * in_zp;
        if (std::abs((int)bias_int16_val - tmp_zp) >
            std::numeric_limits<int16_t>::max()) {
          overflow_ratio = 1;
        }
      }
    } while (overflow_ratio > 0.03 && rightShiftTmp > 0);
    lshift_int8->data()[cidx] = -rightShiftTmp;

    // quantize bias
    scale_val = std::pow(2, rightShiftTmp) / out_scale;
    quantizeToInt16(bias_f32->data() + cidx, bias_int16->data() + cidx, 1,
                    scale_val);
    if (asymmetric) {
      scale_val = std::pow(2, rightShiftTmp) * in_scale / out_scale;
      int8_t tmp_scale = 1;
      quantizeToInt8(scale_f32->data() + cidx, &tmp_scale, 1, scale_val);
      int tmp_zp = tmp_scale * in_zp;
      bias_int16->data()[cidx] -= tmp_zp;
    }
    // due to floor round mode in shift operation, add 1 << (rightShiftTmp - 1)
    // to bias to change the floor to nearest round mode
    if (rightShiftTmp > 0) {
      int16_t bias_shift =
          bias_int16->data()[cidx] + (1 << (rightShiftTmp - 1));
      bias_int16->data()[cidx] = std::max(bias_shift, bias_int16->data()[cidx]);
    }

    // quantize scale
    scale_val = std::pow(2, rightShiftTmp) * in_scale / out_scale;
    quantizeToInt8(scale_f32->data() + cidx, scale_int8->data() + cidx, 1,
                   scale_val);
  }

  // scale
  auto scale_type = op.getScale().getType().cast<RankedTensorType>();
  auto new_scale_type =
      RankedTensorType::get(scale_type.getShape(), rewriter.getI8Type());
  auto new_scale =
      top::WeightOp::create(op, "scale_int8", *scale_int8, new_scale_type);

  // bias
  auto new_bias_type =
      RankedTensorType::get({1, c, 1, 1}, rewriter.getIntegerType(16, true));
  auto new_bias =
      top::WeightOp::create(op, "bias_int16", *bias_int16, new_bias_type);

  // lshift
  auto new_lshift_type =
      RankedTensorType::get({1, c, 1, 1}, rewriter.getI8Type());
  auto new_lshift =
      top::WeightOp::create(op, "lshift_i8", *lshift_int8, new_lshift_type);

  operands.push_back(op.getInput());
  operands.push_back(new_scale);
  operands.push_back(new_bias);
  operands.push_back(new_lshift);

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::ScaleOp>(op, newType, operands, attrs);
}

void ScaleLowering::LoweringBF16(PatternRewriter &rewriter,
                                 top::ScaleOp op) const {
  // lowering_common_bf16<tpu::ScaleOp>(rewriter, op);
  auto ctx = op->getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(op.getOperand(i).getDefiningOp())) {
      operands.push_back(constOp.clone_bf16(op));
    } else {
      operands.push_back(op->getOperand(i));
    }
  }
  // lshift
  auto none = module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto op_name = module::getName(op.getOperation()).str();
  auto newType = getQuantBF16Type(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::ScaleOp>(op, newType, operands, attrs);
}

void ScaleLowering::LoweringF16(PatternRewriter &rewriter,
                                top::ScaleOp op) const {
  // lowering_common_f16<tpu::ScaleOp>(rewriter, op);
  auto ctx = op->getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    if (auto constOp =
            dyn_cast<top::WeightOp>(op.getOperand(i).getDefiningOp())) {
      operands.push_back(constOp.clone_f16(op));
    } else {
      operands.push_back(op->getOperand(i));
    }
  }
  // lshift
  auto none = module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto op_name = module::getName(op.getOperation()).str();
  auto newType = getQuantF16Type(op->getResult(0));
  rewriter.replaceOpWithNewOp<tpu::ScaleOp>(op, newType, operands, attrs);
}

void ScaleLowering::LoweringF8(PatternRewriter &rewriter,
                               top::ScaleOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void ScaleLowering::LoweringQuantized(PatternRewriter &rewriter,
                                      top::ScaleOp op) const {
  lowering_common<tpu::ScaleOp>(rewriter, op, op.getOutput().getType());
}

} // namespace bm1684x
} // namespace tpu_mlir
