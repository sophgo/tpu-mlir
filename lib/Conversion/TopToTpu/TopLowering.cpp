//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/TopToTpu.h"

namespace tpu_mlir {

MLIRContext *LoweringConfig::context;
std::string LoweringConfig::chip;
std::string LoweringConfig::mode;
bool LoweringConfig::isAsymmetric;
std::map<std::string, llvm::StringRef> LoweringConfig::quantize_map;

Value do_transfer(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto in_shape = Module::getShape(in);
  auto out_type = Quant::getQuantInt8Type(out, asymmetric);
  auto ele_type = out_type.cast<RankedTensorType>().getElementType();
  auto new_type = RankedTensorType::get(in_shape, ele_type);

  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = Module::getName(in.getDefiningOp());
  auto out_name = Module::getName(op);
  auto new_name = in_name + "_to_" + out_name;
  int multiplier, rshift;
  get_scale_and_shift(in_scale / out_scale, multiplier, rshift, 8);
  if (in_zp == 0 && out_zp == 0) {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    auto in_type = in.getType().cast<RankedTensorType>();
    auto in_shape = in_type.getShape();
    builder.setInsertionPointAfterValue(in);
    auto mrOp = builder.create<tpu::MulShiftOp>(name_loc, new_type,
                                                ValueRange{in}, attrs);
    return mrOp.output();
  } else {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    attrs.push_back(builder.getNamedAttr(
        "quant_mode",
        tpu::RequantModeAttr::get(op->getContext(), tpu::RequantMode::Normal)));
    builder.setInsertionPointAfterValue(in);
    auto rqOp = builder.create<tpu::RequantIntOp>(name_loc, new_type,
                                                  ValueRange{in}, attrs);
    return rqOp.output();
  }
}

Value do_transfer_fp(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = Module::getName(in.getDefiningOp());
  auto in_stype = Module::getStorageType(in);
  float offset = out_zp;
  auto in_shape = Module::getShape(in);
  auto rq_in = in;
  if (in_stype.isInteger(8) || in_zp != 0 && out_zp != 0) {
    auto add_name = in_name + "_add_zp";
    auto add_type = RankedTensorType::get(in_shape, builder.getI32Type());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("const_val", builder.getF64FloatAttr(in_zp)));
    auto name_loc = NameLoc::get(builder.getStringAttr(add_name));
    auto addOp = builder.create<tpu::AddConstOp>(name_loc, add_type,
                                                 ValueRange{in}, attrs);
    rq_in = addOp.output();
  } else if (in_zp != 0 && out_zp == 0) {
    offset = in_scale / out_scale * (-in_zp);
  }

  auto out_name = Module::getName(op);
  auto new_name = in_name + "_to_" + out_name;

  auto rq_stype = Module::getElementType(out);
  auto rq_type = RankedTensorType::get(in_shape, rq_stype);
  std::vector<NamedAttribute> attrs;
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  attrs.push_back(builder.getNamedAttr(
      "scale", builder.getF64FloatAttr(in_scale / out_scale)));
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getF64FloatAttr(offset)));
  attrs.push_back(builder.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(op->getContext(), tpu::RequantMode::Normal)));
  auto rqOp = builder.create<tpu::RequantFpOp>(name_loc, rq_type,
                                               ValueRange{rq_in}, attrs);
  if (out_zp == 0) {
    return rqOp.output();
  } else {
    llvm_unreachable("Not support now.\n");
  }
}

Value do_dequant(Value input, Type to_type, int64_t multiplier, int64_t shift,
                 tpu::DequantMode mode, int64_t lshift) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(Module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("shift", builder.getI64IntegerAttr(shift)));
  if (mode == tpu::DequantMode::TFlite) {
    attrs.push_back(
        builder.getNamedAttr("lshift", builder.getI64IntegerAttr(lshift)));
  }
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::DequantModeAttr::get(ctx, mode)));

  std::string new_name =
      Module::getName(input.getDefiningOp()).str() + "_dequant";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp = builder.create<tpu::DequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.output();
}

Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(Module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(-shift)));
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp = builder.create<tpu::RequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.output();
}

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands = {input, quant};

  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(Module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp =
      builder.create<tpu::RequantIntAxisOp>(name_loc, newType, operands, attrs);
  return newOp.output();
}

Value do_add_zp(Value input, Type to_type, int64_t zero_point) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(Module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("const_val", builder.getF64FloatAttr(zero_point)));

  std::string new_name =
      Module::getName(input.getDefiningOp()).str() + "_add_zp";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp = builder.create<tpu::AddConstOp>(name_loc, newType,
                                               ValueRange{input}, attrs);
  return newOp.output();
}

} // namespace tpu_mlir
