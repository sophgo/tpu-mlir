//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Lowering.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;

Value top::ScaleOp::lowering_int8_bm1684x(bool asymmetric) {
  auto op = getOperation();
  OpBuilder builder(op);
  int64_t n, c, h, w;
  Module::getNCHW(output(), n, c, h, w);

  std::vector<Value> operands;
  int64_t in_zp, out_zp;
  double in_scale, out_scale;
  Quant::getScaleAndZeroPoint(input(), in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(output(), out_scale, out_zp, asymmetric);

  auto scaleOp = cast<top::WeightOp>(scale().getDefiningOp());
  auto biasOp = cast<top::WeightOp>(bias().getDefiningOp());
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
  auto scale_type = scale().getType().cast<RankedTensorType>();
  auto new_scale_type =
      RankedTensorType::get(scale_type.getShape(), builder.getI8Type());
  auto new_scale =
      WeightOp::create(op, "scale_int8", *scale_int8, new_scale_type);

  // bias
  auto new_bias_type =
      RankedTensorType::get({1, c, 1, 1}, builder.getIntegerType(16, true));
  auto new_bias =
      WeightOp::create(op, "bias_int16", *bias_int16, new_bias_type);

  // lshift
  auto new_lshift_type =
      RankedTensorType::get({1, c, 1, 1}, builder.getI8Type());
  auto new_lshift =
      WeightOp::create(op, "lshift_i8", *lshift_int8, new_lshift_type);

  operands.push_back(input());
  operands.push_back(new_scale);
  operands.push_back(new_bias);
  operands.push_back(new_lshift);

  builder.setInsertionPointAfter(op);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("do_relu", do_reluAttr()));
  attrs.push_back(builder.getNamedAttr("relu_limit", relu_limitAttr()));
  auto newType = Quant::getQuantInt8Type(output(), asymmetric);
  auto newOp =
      builder.create<tpu::ScaleOp>(op->getLoc(), newType, operands, attrs);
  return newOp.output();
}

Value top::ScaleOp::lowering_f32_bm1684x() {
  auto op = getOperation();
  auto ctx = op->getContext();
  auto output = op->getResult(0);
  auto sType = Module::getStorageType(output);
  auto shape = Module::getShape(output);
  Type newType = output.getType();
  if (sType.isa<FloatType>() == false) {
    if (Quant::isCalibratedType(output)) {
      auto caliType = Quant::getCalibratedType(output);
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
  auto none = Module::getNoneOp(op);
  operands.push_back(none);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  builder.setInsertionPointAfter(op);
  auto newOp =
      builder.create<tpu::ScaleOp>(op->getLoc(), newType, operands, attrs);
  return newOp.output();
}

Value top::ScaleOp::lowering_bf16_bm1684x() {
  return lowering_common_float<tpu::ScaleOp, BFloat16Type>(getOperation());
}

Value top::ScaleOp::lowering_f16_bm1684x() {
  return lowering_common_float<tpu::ScaleOp, Float16Type>(getOperation());
}

Value top::ScaleOp::lowering_quant_bm1684x() {
  return lowering_common_float<tpu::ScaleOp>(getOperation());
}
