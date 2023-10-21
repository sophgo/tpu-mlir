//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/Conversion.h"

namespace tpu_mlir {

bool LoweringConfig::isQuantized;
bool LoweringConfig::doWinograd;
std::map<std::string, module::Mode> LoweringConfig::quantize_map;

Value do_transfer(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (module::isSign(in) == module::isSign(out) && in_scale == out_scale &&
      in_zp == out_zp) {
    return in;
  }
  auto in_shape = module::getShape(in);
  auto out_type = getQuantInt8Type(out, asymmetric);
  auto ele_type = out_type.cast<RankedTensorType>().getElementType();
  auto new_type = RankedTensorType::get(in_shape, ele_type);

  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = module::getName(in.getDefiningOp());
  auto out_name = module::getName(op);
  auto new_name = in_name.str() + "_to_" + out_name.str();
  int multiplier, rshift;
  get_scale_and_shift_positive(in_scale / out_scale, multiplier, rshift, 8);
  if (in_zp == 0 && out_zp == 0) {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getSI32IntegerAttr(rshift)));
    builder.setInsertionPointAfterValue(in);
    auto mrOp = builder.create<tpu::MulShiftOp>(name_loc, new_type,
                                                ValueRange{in}, attrs);
    return mrOp.getOutput();
  } else {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getSI32IntegerAttr(rshift)));
    attrs.push_back(builder.getNamedAttr(
        "quant_mode",
        tpu::RequantModeAttr::get(op->getContext(),
                                  tpu::RequantMode::MultiplierShift)));
    builder.setInsertionPointAfterValue(in);
    auto rqOp = builder.create<tpu::RequantIntOp>(name_loc, new_type,
                                                  ValueRange{in}, attrs);
    return rqOp.getOutput();
  }
}

Value do_transfer_fp(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  module::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  module::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = module::getName(in).str();
  auto in_stype = module::getStorageType(in);
  float offset = out_zp;
  auto in_shape = module::getShape(in);
  auto rq_in = in;
  if (in_stype.isInteger(8) || (in_zp != 0 && out_zp != 0)) {
    auto add_name = in_name + "_add_zp";
    auto add_type = RankedTensorType::get(in_shape, builder.getI32Type());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("const_val", builder.getF64FloatAttr(in_zp)));
    auto name_loc = NameLoc::get(builder.getStringAttr(add_name));
    auto addOp = builder.create<tpu::AddConstOp>(name_loc, add_type,
                                                 ValueRange{in}, attrs);
    rq_in = addOp.getOutput();
  } else if (in_zp != 0 && out_zp == 0) {
    offset = in_scale / out_scale * (-in_zp);
  }

  auto out_name = module::getName(op).str();
  auto new_name = in_name + "_to_" + out_name;

  auto rq_type = module::getTypeLike(out, in_shape);
  std::vector<NamedAttribute> attrs;
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  attrs.push_back(builder.getNamedAttr(
      "scale", builder.getF64FloatAttr(in_scale / out_scale)));
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getF64FloatAttr(offset)));
  attrs.push_back(builder.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(
                        op->getContext(), tpu::RequantMode::MultiplierShift)));
  auto rqOp = builder.create<tpu::RequantFpOp>(name_loc, rq_type,
                                               ValueRange{rq_in}, attrs);
  if (out_zp == 0) {
    return rqOp.getOutput();
  } else {
    llvm_unreachable("Not support now.\n");
  }
}

Value do_dequant(Location name_loc, Value input, Type to_type,
                 int64_t multiplier, int64_t shift, tpu::DequantMode mode,
                 int64_t lshift) {
  [[maybe_unused]]auto from_stype = module::getStorageType(input);
  [[maybe_unused]]auto to_stype = module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("shift", builder.getI64IntegerAttr(shift)));
  if (mode == tpu::DequantMode::TFLite) {
    attrs.push_back(
        builder.getNamedAttr("lshift", builder.getI64IntegerAttr(lshift)));
  }
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::DequantModeAttr::get(ctx, mode)));

  auto newOp = builder.create<tpu::DequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.getOutput();
}

Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode) {
  auto to_stype = module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getSI32IntegerAttr(-shift)));
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp = builder.create<tpu::RequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.getOutput();
}

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode) {
  [[maybe_unused]]auto from_stype = module::getStorageType(input);
  auto to_stype = module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands = {input, quant};

  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp =
      builder.create<tpu::RequantIntAxisOp>(name_loc, newType, operands, attrs);
  return newOp.getOutput();
}

Value do_requantFp(Value input, double scale, double offset, Type to_type,
                   std::string &to_name, tpu::RequantMode mode) {
  [[maybe_unused]]auto from_stype = module::getStorageType(input);
  [[maybe_unused]]auto ctx = input.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(input);
  auto in_name = module::getName(input).str() + "_" + to_name;
  std::vector<NamedAttribute> attrs;
  auto name_loc = NameLoc::get(builder.getStringAttr(in_name));
  attrs.push_back(
      builder.getNamedAttr("scale", builder.getF64FloatAttr(scale)));
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getF64FloatAttr(offset)));
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));
  auto rqOp = builder.create<tpu::RequantFpOp>(name_loc, to_type,
                                               ValueRange{input}, attrs);

  return rqOp.getOutput();
}

Value do_reshape(Value input, RankedTensorType to_type) {
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs = {};
  std::string new_name =
      module::getName(input.getDefiningOp()).str() + "_reshape";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp = builder.create<tpu::ReshapeOp>(name_loc, to_type,
                                              ValueRange{input}, attrs);
  return newOp.getOutput();
}

int32_t do_const_dequant(Value input, int64_t multiplier, int64_t shift,
                         int64_t lshift) {
  auto qtype = module::getUniformQuantizedType(input);
  auto input_stype = module::getStorageType(input);
  auto input_quant = cast<top::WeightOp>(input.getDefiningOp()).read<int8_t>();
  int64_t input_offset = qtype.getZeroPoint();
  int32_t input_data = input_stype.isUnsignedInteger(8)
                           ? (uint8_t)(input_quant->at(0))
                           : input_quant->at(0);
  int64_t tmp = (input_data - input_offset) * (int64_t)multiplier;
  auto v = RightShiftRound(tmp, 31 - lshift, ROUNDING_HALF_UP);
  v = RightShiftRound(v, shift, ROUNDING_HALF_AWAY_FROM_ZERO);
  return v;
}

Value do_weight_dequant(Value input, Type to_type, int64_t multiplier,
                        int64_t shift, int64_t lshift) {
  auto op = input.getDefiningOp();
  auto qtype = module::getUniformQuantizedType(input);
  auto input_stype = module::getStorageType(input);
  int64_t num_elem = module::getNumElements(input);
  auto input_dequant = std::make_shared<std::vector<int32_t>>(num_elem);
  auto input_quant = cast<top::WeightOp>(input.getDefiningOp()).read<int8_t>();
  int64_t input_offset = qtype.getZeroPoint();
  if (input_stype.isUnsignedInteger(8)) {
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int64_t tmp = ((uint8_t)(input_quant->at(idx)) - input_offset) *
                    (int64_t)multiplier;
      auto v = RightShiftRound(tmp, 31 - lshift, ROUNDING_HALF_UP);
      v = RightShiftRound(v, -shift, ROUNDING_HALF_AWAY_FROM_ZERO);
      input_dequant->data()[idx] = v;
    }
  } else {
    for (int64_t idx = 0; idx < num_elem; idx++) {
      int64_t tmp = (input_quant->at(idx) - input_offset) * (int64_t)multiplier;
      auto v = RightShiftRound(tmp, 31 - lshift, ROUNDING_HALF_UP);
      v = RightShiftRound(v, -shift, ROUNDING_HALF_AWAY_FROM_ZERO);
      input_dequant->data()[idx] = v;
    }
  }
  auto new_type = RankedTensorType::get(module::getShape(input), to_type);
  return top::WeightOp::create(op, "_dequant", *input_dequant, new_type);
}

Type getQuantInt8Type(Value v, bool asymmetric) {
  if (module::isNone(v)) {
    return v.getType();
  }
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = module::getCalibratedType(v);
  auto min = cali_type.getMin();
  double scale;
  int64_t zeropoint = 0;
  module::getScaleAndZeroPoint(v, scale, zeropoint, asymmetric);
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (min >= 0) {
    qmin = 0;
    qmax = 255;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

Type getQuantIntType(Value v, double scale, double offset, int bits) {
  assert(bits == 8 || bits == 4);
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = module::getCalibratedType(v);
  auto min = cali_type.getMin();
  int64_t qmin = -128, qmax = 127;
  if (bits == 4) {
    qmin = -8;
    qmax = 7;
  }
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (min >= 0) {
    qmin = 0;
    qmax = 255;
    if (bits == 4)
      qmax = 15;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(
      flag, IntegerType::get(ctx, bits), cali_type.getExpressedType(), scale,
      offset, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

Type getQuantInt4Type(Value v, bool asymmetric) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = module::getCalibratedType(v);
  auto min = cali_type.getMin();
  double scale;
  int64_t zeropoint = 0;
  int bitwidth = 4;
  module::getScaleAndZeroPoint(v, scale, zeropoint, asymmetric, bitwidth);
  int64_t qmin = -8, qmax = 7;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (min >= 0) {
    qmin = 0;
    qmax = 15;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 4),
                                                cali_type.getExpressedType(),
                                                scale, zeropoint, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

Type getQuantBoolType(Value v) {
  auto type = v.getType().cast<RankedTensorType>();
  auto ctx = v.getContext();
  auto cali_type = module::getCalibratedType(v);
  int64_t qmin = -128, qmax = 127;
  uint32_t flag = quant::QuantizationFlags::Signed;
  if (cali_type.getMin() >= 0) {
    qmin = 0;
    qmax = 255;
    flag = 0;
  }
  auto qtype = quant::UniformQuantizedType::get(flag, IntegerType::get(ctx, 8),
                                                cali_type.getExpressedType(),
                                                1.0, 0, qmin, qmax);
  return RankedTensorType::get(type.getShape(), qtype);
}

Value do_transpose(Location name_loc, Value input,
                   std::vector<int64_t> &order) {
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(input);
  auto inshape = module::getShape(input);
  auto dims = inshape.size();
  std::vector<int64_t> oshape;
  for (int i = 0; i < dims; ++i) {
    oshape.push_back(inshape[order[i]]);
  }
  auto new_type = module::getTypeLike(input, oshape);
  std::vector<NamedAttribute> attrs = {};
  attrs.push_back(
      builder.getNamedAttr("order", builder.getI64ArrayAttr(order)));
  // std::string new_name =
  //     module::getName(input.getDefiningOp()).str() + "_transpose";
  // auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp = builder.create<tpu::PermuteOp>(
      name_loc, new_type,
      ValueRange{input, module::getNoneOp(input.getDefiningOp())}, attrs);
  return newOp.getOutput();
}

Value insert_host2device(Value v, Type to) {
  auto ctx = v.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(v);
  auto name = module::getName(v).str();
  name += "_host2device";
  auto newType = RankedTensorType::get(module::getShape(v), module::getStorageType(v));
  auto loc = NameLoc::get(builder.getStringAttr(name));
  auto hdOp = builder.create<tpu::Host2DeviceOp>(loc, newType, ValueRange{v});
  return hdOp.getOutput();
}

Value insert_device2host(Value v, Type to, Operation* user) {
  auto ctx = v.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointAfterValue(v);
  auto name = module::getName(v).str();
  if (user && !isa<ReturnOp>(user)) {
      name += "_" + module::getName(user).str();
  }
  name += "_device2host";
  auto newType = RankedTensorType::get(module::getShape(v), module::getStorageType(v));
  auto loc = NameLoc::get(builder.getStringAttr(name));
  auto hdOp = builder.create<tpu::Device2HostOp>(loc, newType, ValueRange{v});
  return hdOp.getOutput();
}

void try_insert_host2device(Operation* op, uint32_t idx) {
  auto opd = op->getOperand(idx);
  auto def_op = opd.getDefiningOp();
  if (def_op->hasTrait<trait::ShapeProducer>()) {
    auto hdOp = insert_host2device(opd, opd.getType());
    op->setOperand(idx, hdOp);
  }
}

void try_insert_device2host(Operation* op, uint32_t idx) {
  auto opd = op->getOperand(idx);
  auto def_op = opd.getDefiningOp();
  if (!def_op->hasTrait<trait::ShapeProducer>()) {
    auto hdOp = insert_device2host(opd, opd.getType(), op);
    op->setOperand(idx, hdOp);
  }
}

} // namespace tpu_mlir
