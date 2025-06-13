//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Support/Float8.h"

namespace tpu_mlir {
namespace bm1684x {

static Value CreateConvOp(PatternRewriter &rewriter, int64_t dims, Location loc,
                          Type type, std::vector<Value> &operands,
                          std::vector<NamedAttribute> &attrs) {
  switch (dims) {
  case 1:
  case 2: {
    auto newOp = rewriter.create<tpu::Conv2DOp>(loc, type, operands, attrs);
    return newOp.getOutput();
  } break;
  case 3: {
    auto newOp = rewriter.create<tpu::Conv3DOp>(loc, type, operands, attrs);
    return newOp.getOutput();
  } break;
  default:
    llvm_unreachable("not support kernel dims");
  }
  return nullptr;
}

void ConvLowering::LoweringF32(PatternRewriter &rewriter,
                               top::ConvOp op) const {
  rewriter.setInsertionPointAfter(op);
  auto p = op.parseParam();
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  for (auto i = 0; i < nInputs; ++i) {
    operands.push_back(op->getOperand(i));
  }
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newValue = CreateConvOp(rewriter, p.dims, op->getLoc(),
                               op.getOutput().getType(), operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringINT8(PatternRewriter &rewriter, top::ConvOp op,
                                bool asymmetric) const {
  if (module::isWeight(op.getFilter()) == false) {
    if (module::isMARS3() || module::isSGTPUV8())
      LoweringBF16(rewriter, op);
    else
      LoweringF32(rewriter, op);
    return;
  }

  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  auto p = op.parseParam();
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  bool input_asymmetric = op->hasAttr("input_asym");
  bool output_int16 =
      op->hasAttr("output_int16") && op.getKernelShape().size() == 2;
  module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp,
                               input_asymmetric || asymmetric);
  module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
  if (output_int16) {
    out_scale = out_scale * 255 / 65535;
    out_zp = 0;
  }
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  auto filter_size = filter_f32->size();
  float fmax, fmin;
  findMinMax(filter_f32->data(), filter_size, &fmin, &fmax);
  bool fsign = (fmin < 0 || p.has_bias == true);
  float times = 1;
  bool all_i8 = to_all_int8(*filter_f32, times, fsign);
  float fqmax = fsign ? 127 : 255;
  f64_array_t weight_scale_v;
  if (filterOp.getScale().has_value()) {
    weight_scale_v = module::getF64Array(filterOp.getScale().value());
    all_i8 = false;
  }

  i32_array_t bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_size);
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_size);
  float bias_max = 1e-6;
  if (p.has_bias) {
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    float max = findMaxabs(bias_fp32->data(), bias_fp32->size());
    bias_max = max;
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(p.oc, 0);
  }

  if (in_scale <=
      1e-5) { // if the input is all zero, in th would be 1e-5 in calibration,
              // and the bias may be overflow in quant. if no bias or bias too
              // small, follow filter. occur in sam model from xm.
    in_scale = bias_max <= 1e-5
                   ? std::max(std::abs(fmax), std::abs(fmin)) / 127.0
                   : bias_max / 127.0;
    llvm::errs()
        << module::getName(op.getOperation()).str()
        << " : found input too small in conv lowering int8, scale change to "
        << in_scale << " bias is: " << bias_max << "\n";
  }

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  double scale_w;
  int int32_multiplier, shift;
  int inner_dim = filter_size / p.oc;
  for (int c = 0; c < p.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    if (all_i8) {
      scale_w = 1.0 / times;
    } else if (filterOp.getScale().has_value() && weight_scale_v->size()) {
      scale_w = weight_scale_v->data()[c];
    } else {
      float w_max = findMaxabs(p_filter, inner_dim);
      scale_w = std::max(w_max / fqmax, 1e-5f);
    }
    double scale_f = scale_w * in_scale / out_scale;
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);

    if (fsign) {
      for (int t = 0; t < inner_dim; t++) {
        filter_i8->data()[c * inner_dim + t] = to_int8(p_filter[t] / scale_w);
      }
    } else {
      for (int t = 0; t < inner_dim; t++) {
        filter_u8->data()[c * inner_dim + t] = to_uint8(p_filter[t] / scale_w);
      }
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < inner_dim; t++) {
        bias_w_xz += filter_i8->data()[c * inner_dim + t] * in_zp;
      }
    }

    if (p.has_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  bool has_bias = (bias_int32 != nullptr);

  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, fsign));
  if (fsign) {
    auto new_filter =
        top::WeightOp::create(op, "filter_i8", *filter_i8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter =
        top::WeightOp::create(op, "filter_u8", *filter_u8, new_type);
    operands.push_back(new_filter);
  }
  if (has_bias) {
    std::vector<int64_t> bias_shape(module::getShape(op.getOutput()).size(), 1);
    bias_shape[1] = p.oc;
    auto new_type = RankedTensorType::get(bias_shape, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(op.getBias()); // none
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(has_bias)));

  auto ctx = op->getContext();
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
  attrs.push_back(rewriter.getNamedAttr(
      "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
  attrs.push_back(rewriter.getNamedAttr(
      "multiplier", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));
  Type newType;
  if (output_int16) {
    newType = getQuantIntType(op.getOutput(), out_scale, out_zp, 16);
  } else {
    newType = getQuantInt8Type(op.getOutput(), asymmetric);
  }

  bool output_int32 = false;
  if (op.getKernelShape().size() == 3) {
    output_int32 = true;
  }

  if (output_int32) {
    // to int32, and then requant to int8
    auto convType = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getI32Type());
    auto conv_name = module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(conv_name));
    auto conv_out = CreateConvOp(rewriter, op.getKernelShape().size(), name_loc,
                                 convType, operands, attrs);
    // requant
    auto output_type = getQuantInt8Type(op.getOutput(), asymmetric);
    std::vector<int32_t> quant;
    int64_t quant_w_size = 0;
    if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
        module::isSGTPUV8()) {
      quant_w_size = 2;
      quant.resize(p.oc * quant_w_size, 0);
      for (size_t i = 0; i < p.oc; ++i) {
        quant[i * 2] = multiplier_v[i];
        quant[i * 2 + 1] = ((-(int32_t)rshift_v[i]) & 0xffff) |
                           (((int32_t)out_zp & 0xffff) << 16);
      }
    } else {
      quant_w_size = 3;
      quant.resize(p.oc * quant_w_size, 0);
      for (size_t i = 0; i < p.oc; ++i) {
        quant[i * 3] = multiplier_v[i];
        quant[i * 3 + 1] = -rshift_v[i];
        quant[i * 3 + 2] = out_zp;
      }
    }
    auto quant_type = RankedTensorType::get({1, p.oc, 1, 1, quant_w_size},
                                            rewriter.getI32Type());
    auto quant_value = top::WeightOp::create(op, "quant", quant, quant_type);
    auto newValue = do_requant(op->getLoc(), conv_out, quant_value, output_type,
                               true, tpu::RequantMode::MultiplierShift);
    rewriter.replaceOp(op, {newValue});
    return;
  }

  auto newValue =
      CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
  return;
}

void ConvLowering::LoweringINT4(PatternRewriter &rewriter, top::ConvOp op,
                                bool asymmetric) const {
  llvm::errs() << "start conv LoweringINT4, name:"
               << module::getName(op.getOperation()).str() << "\n";
  auto p = op.parseParam();
  if (p.is_dw /*|| p.sw > 1*/) {
    return LoweringINT8(rewriter, op, asymmetric);
  }
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;

  // in/out scale/zp
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  double in_int8_scale;
  int64_t in_int8_zp;
  int bitwidth = 4;
  Value value;
  if (op.getInInt4Scale().has_value()) {
    // bool find = false; //会导致性能大幅下降，原因待分析
    // for (auto user : op.getInput().getDefiningOp()->getUsers()) {
    //   if (isa<tpu::RequantFpOp>(user)) {
    //     find = true;
    //     operands.push_back(user->getResults()[0]);
    //     break;
    //   }
    // }
    // if (!find) {
    // 存在int4的输入scale，说明上一层是int8，故输入tensor也是int8，需要requant为int4
    in_scale = op.getInInt4Scale().value().convertToDouble();
    in_zp = op.getInInt4Zp().value().convertToDouble();
    module::getScaleAndZeroPoint(op.getInput(), in_int8_scale, in_int8_zp,
                                 asymmetric);
    auto output_type = getQuantIntType(op.getInput(), in_scale, in_zp, 4);
    double scale = in_int8_scale / in_scale; // 将int8转为int4的rq参数
    double offset = in_zp - in_int8_zp * scale;
    auto to_name = "to_b4_for_" + module::getName(op.getOperation()).str();
    value = do_requantFp(op.getInput(), scale, offset, output_type, to_name);
    llvm::errs() << "conv input requantFp, to_name:" << to_name << "\n";
    value.dump();
    operands.push_back(value);
    // }
  } else { // 输入tensor也是int4
    operands.push_back(op.getInput());
    module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric,
                                 bitwidth);
  }
  module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric,
                               bitwidth);
  // filter
  auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
  auto filter_f32 = filterOp.read<float>();
  float fmax, fmin;
  findMinMax(filter_f32->data(), filter_f32->size(), &fmin, &fmax);
  bool fsign = (fmin < 0 || p.has_bias == true);
  float fqmax = fsign ? 7 : 15;
  f64_array_t weight_scale_v;
  if (filterOp.getScale().has_value()) {
    weight_scale_v = module::getF64Array(filterOp.getScale().value());
  }

  i32_array_t bias_int32;
  std::shared_ptr<std::vector<float>> bias_fp32;
  auto filter_i8 = std::make_shared<std::vector<int8_t>>(filter_f32->size());
  auto filter_u8 = std::make_shared<std::vector<uint8_t>>(filter_f32->size());
  if (p.has_bias) {
    auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
    bias_fp32 = biasOp.read<float>();
    bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
  } else if (in_zp) {
    bias_int32 = std::make_shared<std::vector<int32_t>>(p.oc, 0);
  }

  bool all_next_layer_is_int8 = true;
  bool all_next_layer_is_int4 = true;
  double out_int8_scale =
      op.getOutInt8Scale().value_or(APFloat(1.0)).convertToDouble();
  double out_int8_zp =
      op.getOutInt8Zp().value_or(APFloat(0.0)).convertToDouble();
  for (auto user : op->getUsers()) {
    if (module::isInt4Op(user)) {
      all_next_layer_is_int8 = false;
    } else {
      all_next_layer_is_int4 = false;
    }
  }

  llvm::errs() << "all_next_layer_is_int4:" << all_next_layer_is_int4
               << ",all_next_layer_is_int8:" << all_next_layer_is_int8 << "\n";
  if (all_next_layer_is_int8)
    llvm::errs() << "directly output int8\n";
  else
    llvm::errs() << "directly output int4\n";

  std::vector<int64_t> rshift_v;
  std::vector<int64_t> multiplier_v;
  double scale_w;
  int int32_multiplier, shift;
  int inner_dim = filter_f32->size() / p.oc;
  for (int c = 0; c < p.oc; c++) { // per-channel量化
    float *p_filter = filter_f32->data() + c * inner_dim;
    if (filterOp.getScale().has_value() && weight_scale_v->size()) {
      scale_w = weight_scale_v->data()[c];
    } else {
      float w_max = findMaxabs(p_filter, inner_dim);
      scale_w = std::max(w_max / fqmax, 1e-5f);
    }
    double scale_f;
    if (all_next_layer_is_int8) {
      scale_f = scale_w * in_scale / out_int8_scale;
    } else {
      scale_f = scale_w * in_scale / out_scale;
    }
    get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
    multiplier_v.push_back(int32_multiplier);
    rshift_v.push_back(shift);
    if (fsign) {
      for (int t = 0; t < inner_dim; t++) {
        filter_i8->data()[c * inner_dim + t] = to_int4(p_filter[t] / scale_w);
      }
    } else {
      for (int t = 0; t < inner_dim; t++) {
        filter_u8->data()[c * inner_dim + t] = to_uint4(p_filter[t] / scale_w);
      }
    }

    double bias_w_xz = 0;
    if (in_zp) {
      for (int t = 0; t < inner_dim; t++) {
        bias_w_xz += filter_i8->data()[c * inner_dim + t] * in_zp;
      }
    }

    if (p.has_bias) {
      bias_int32->data()[c] =
          std::round(bias_fp32->data()[c] / (scale_w * in_scale) - bias_w_xz);
    } else if (in_zp) {
      bias_int32->data()[c] = std::round(-bias_w_xz);
    }
  }
  bool has_bias = (bias_int32 != nullptr);

  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto new_type = RankedTensorType::get(filter_type.getShape(),
                                        rewriter.getIntegerType(8, fsign));
  if (fsign) {
    auto new_filter =
        top::WeightOp::create(op, "filter_i4", *filter_i8, new_type);
    operands.push_back(new_filter);
  } else {
    auto new_filter =
        top::WeightOp::create(op, "filter_u4", *filter_u8, new_type);
    operands.push_back(new_filter);
  }
  if (has_bias) {
    std::vector<int64_t> bias_shape(module::getShape(op.getOutput()).size(), 1);
    bias_shape[1] = p.oc;
    auto new_type = RankedTensorType::get(bias_shape, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
    operands.push_back(new_bias);
  } else {
    operands.push_back(op.getBias()); // none
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(has_bias)));

  bool output_int32 = false;
  if (op.getKernelShape().size() == 3) {
    output_int32 = true;
  }
  if (output_int32) {
    // to int32, and then requant to int8
    auto convType = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getI32Type());
    auto conv_name = module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(conv_name));
    auto conv_out = CreateConvOp(rewriter, op.getKernelShape().size(), name_loc,
                                 convType, operands, attrs);
    // requant
    auto output_type = getQuantInt4Type(op.getOutput(), asymmetric);
    std::vector<int32_t> quant;
    int64_t quant_w_size = 0;
    if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
        module::isSGTPUV8()) {
      quant_w_size = 2;
      quant.resize(p.oc * quant_w_size, 0);
      for (size_t i = 0; i < p.oc; ++i) {
        quant[i * 2] = multiplier_v[i];
        quant[i * 2 + 1] = ((-(int32_t)rshift_v[i]) & 0xffff) |
                           (((int32_t)out_zp & 0xffff) << 16);
      }
    } else {
      quant_w_size = 3;
      quant.resize(p.oc * quant_w_size, 0);
      for (size_t i = 0; i < p.oc; ++i) {
        quant[i * 3] = multiplier_v[i];
        quant[i * 3 + 1] = -rshift_v[i];
        quant[i * 3 + 2] = out_zp;
      }
    }
    auto quant_type = RankedTensorType::get({1, p.oc, 1, 1, quant_w_size},
                                            rewriter.getI32Type());
    auto quant_value = top::WeightOp::create(op, "quant", quant, quant_type);
    auto newValue = do_requant(op->getLoc(), conv_out, quant_value, output_type,
                               true, tpu::RequantMode::MultiplierShift);
    rewriter.replaceOp(op, {newValue});
    return;
  }

  if (!all_next_layer_is_int8 && !all_next_layer_is_int4) {
    // to int32, and then requant to int8
    auto convType = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getI32Type());
    auto conv_name = module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(conv_name));
    auto conv_out = CreateConvOp(rewriter, op.getKernelShape().size(), name_loc,
                                 convType, operands, attrs);

    std::vector<Operation *> int8_op;
    std::vector<Operation *> int4_op;
    std::vector<Operation *> cur_op;
    for (auto user : op->getUsers()) {
      if (!module::isInt4Op(user)) {
        int8_op.push_back(user);
      } else {
        int4_op.push_back(user);
      }
    }

    auto ctx = op.getOutput().getContext();
    OpBuilder builder(ctx);
    for (int i = 0; i < 2; i++) {
      Type newType;
      std::string w_name, requant_name;
      if (i == 0) {
        w_name = "w_quant_int8_for_" + module::getName(op.getOperation()).str();
        requant_name =
            "requant_int8_for_" + module::getName(op.getOperation()).str();
        cur_op.swap(int8_op);
        newType = getQuantIntType(op.getOutput(), out_int8_scale, out_int8_zp);
      } else {
        w_name = "w_quant_int4_for_" + module::getName(op.getOperation()).str();
        requant_name =
            "requant_int4_for_" + module::getName(op.getOperation()).str();
        cur_op.swap(int4_op);
        newType = getQuantInt4Type(op.getOutput(), asymmetric);
      }
      auto requant_name_loc = NameLoc::get(builder.getStringAttr(requant_name));
      multiplier_v.clear();
      rshift_v.clear();
      for (int c = 0; c < p.oc; c++) { // per-channel量化
        float *p_filter = filter_f32->data() + c * inner_dim;
        if (filterOp.getScale().has_value() && weight_scale_v->size()) {
          scale_w = weight_scale_v->data()[c];
        } else {
          float w_max = findMaxabs(p_filter, inner_dim);
          scale_w = std::max(w_max / fqmax, 1e-5f);
        }
        double scale_f;
        if (i == 0)
          scale_f = scale_w * in_scale / out_int8_scale;
        else
          scale_f = scale_w * in_scale / out_scale;
        get_scale_and_shift(scale_f, int32_multiplier, shift, 32);
        multiplier_v.push_back(int32_multiplier);
        rshift_v.push_back(shift);
      }
      // requant
      std::vector<int32_t> quant;
      int64_t quant_w_size = 0;
      if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
          module::isSGTPUV8()) {
        quant_w_size = 2;
        quant.resize(p.oc * quant_w_size, 0);
        for (size_t i = 0; i < p.oc; ++i) {
          quant[i * 2] = multiplier_v[i];
          quant[i * 2 + 1] = ((-(int32_t)rshift_v[i]) & 0xffff) |
                             (((int32_t)out_zp & 0xffff) << 16);
        }
      } else {
        quant_w_size = 3;
        quant.resize(p.oc * quant_w_size, 0);
        for (size_t i = 0; i < p.oc; ++i) {
          quant[i * 3] = multiplier_v[i];
          quant[i * 3 + 1] = -rshift_v[i];
          quant[i * 3 + 2] = out_zp;
        }
      }
      auto quant_type = RankedTensorType::get({1, p.oc, 1, 1, quant_w_size},
                                              rewriter.getI32Type());
      auto quant_value = top::WeightOp::create(op, w_name, quant, quant_type);

      auto newValue =
          do_requant(requant_name_loc, conv_out, quant_value, newType, true,
                     tpu::RequantMode::MultiplierShift);

      for (auto op2 : cur_op) {
        std::string str = module::getName(op2).str();
        for (uint32_t idx = 0; idx < op2->getNumOperands(); idx++) {
          if (op.getOutput() == op2->getOperand(idx)) {
            llvm::errs() << "setOperand, idx:" << idx << ",name:" << str
                         << "\n";
            op2->setOperand(idx, newValue);
          }
        }
      }
    }
    rewriter.replaceOp(op, {conv_out});
  } else {
    auto ctx = op->getContext();
    attrs.push_back(rewriter.getNamedAttr(
        "quant_mode",
        tpu::RequantModeAttr::get(ctx, tpu::RequantMode::MultiplierShift)));
    attrs.push_back(rewriter.getNamedAttr(
        "rshift", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{rshift_v})));
    attrs.push_back(rewriter.getNamedAttr(
        "multiplier",
        rewriter.getI64ArrayAttr(ArrayRef<int64_t>{multiplier_v})));

    auto newType = getQuantInt4Type(op.getOutput(), asymmetric);
    if (all_next_layer_is_int8) {
      newType = getQuantIntType(op.getOutput(), out_int8_scale, out_int8_zp);
    }
    auto newValue =
        CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);

    rewriter.replaceOp(op, {newValue});
  }

  return;
}

void ConvLowering::LoweringBF16(PatternRewriter &rewriter,
                                top::ConvOp op) const {
  // if (module::isWeight(op.getFilter()) == false) {
  //   LoweringF32(rewriter, op);
  //   return;
  // }
  auto p = op.parseParam();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  for (int i = 0; i < 2; i++) {
    // 0: input; 1: filter, both have possibility to be WeightOp
    auto opd = op.getOperand(i);
    if (auto weightOp = dyn_cast<top::WeightOp>(opd.getDefiningOp())) {
      operands.push_back(weightOp.clone_bf16(op));
    } else {
      operands.push_back(opd);
    }
  }

  operands.push_back(op.getBias());
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantBF16Type(op.getOutput());
  auto newValue =
      CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringF16(PatternRewriter &rewriter,
                               top::ConvOp op) const {
  // if (module::isWeight(op.getFilter()) == false) {
  //   LoweringF32(rewriter, op);
  //   return;
  // }
  auto p = op.parseParam();
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;

  for (int i = 0; i < 2; i++) {
    // 0: input; 1: filter, both have possibility to be WeightOp
    auto opd = op.getOperand(i);
    if (auto weightOp = dyn_cast<top::WeightOp>(opd.getDefiningOp())) {
      operands.push_back(weightOp.clone_f16(op));
    } else {
      operands.push_back(opd);
    }
  }

  // bias lowering will be done in --weight-reorder pass
  auto bias_value = op.getBias();
  bool with_bias = !module::isNone(bias_value);
  if (with_bias && !module::isTrain())
    ASSERT_OP(
        module::getStorageType(bias_value).isF32() && "bias has to be f32", op);
  operands.push_back(bias_value);
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  auto newType = getQuantF16Type(op.getOutput());
  auto newValue =
      CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);
  rewriter.replaceOp(op, {newValue});
}

void ConvLowering::LoweringF8(PatternRewriter &rewriter, top::ConvOp op) const {
  auto p = op.parseParam();
  double in_scale = 1.0, out_scale = 1.0;
  auto in = op.getInput();
  auto out = op.getOutput();

  if (module::getMode() == module::Mode::F8E4M3) {
    auto qtype_in = module::getCalibratedType(in);
    auto qtype_out = module::getCalibratedType(out);
    in_scale = qtype_in.getMax() / get_f8e4m3_max();
    out_scale = qtype_out.getMax() / get_f8e4m3_max();
  } else if (module::getMode() == module::Mode::F8E5M2) {
    in_scale = 1.0;
    out_scale = 1.0;
  }

  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  Value newWeight;
  for (int i = 0; i < 2; i++) {
    // 0: input; 1: filter, both have possibility to be WeightOp
    auto opd = op.getOperand(i);
    if (auto weightOp = dyn_cast<top::WeightOp>(opd.getDefiningOp())) {
      if (module::getMode() == module::Mode::F8E4M3) {
        newWeight = weightOp.clone_f8e4m3(op, true, true);
        operands.push_back(newWeight);
      } else if (module::getMode() == module::Mode::F8E5M2) {
        newWeight = weightOp.clone_f8e5m2(op);
        operands.push_back(newWeight);
      }
    } else {
      operands.push_back(opd);
    }
  }
  f64_array_t weight_scale_v;
  f64_array_t quant_scale_v;
  auto w_op = dyn_cast<top::WeightOp>(newWeight.getDefiningOp());
  if (module::getMode() == module::Mode::F8E4M3) {
    if (!w_op.getScale().has_value())
      llvm_unreachable("weight should has scale now");
    weight_scale_v = module::getF64Array(w_op.getScale().value());
    quant_scale_v =
        std::make_shared<std::vector<double>>(weight_scale_v.get()->size());
    for (int i = 0; i < weight_scale_v.get()->size(); i++) {
      quant_scale_v.get()->at(i) = in_scale * weight_scale_v.get()->at(i);
    }
  } else {
    quant_scale_v = std::make_shared<std::vector<double>>(1, 1.0);
    weight_scale_v = std::make_shared<std::vector<double>>(1, 1.0);
  }

  operands.push_back(op.getBias());
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(op->getContext(),
                                              tpu::RequantMode::OnlyScale)));
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  bool with_bias = !module::isNone(op.getBias());

  if (with_bias && module::getMode() == module::Mode::F8E4M3) {
    auto b_op = dyn_cast<top::WeightOp>(op.getBias().getDefiningOp());
    auto b_value = b_op.read<float>();
    for (int i = 0; i < b_value->size(); i++)
      b_value->at(i) =
          b_value->at(i) / (in_scale * weight_scale_v.get()->at(i));
    b_op.update(*b_value, b_value.get()->size());
  }
  if (module::getMode() == module::Mode::F8E4M3) {
    for (int i = 0; i < weight_scale_v.get()->size(); i++) {
      quant_scale_v.get()->at(i) = quant_scale_v.get()->at(i) / out_scale;
    }
  }

  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  if (module::getMode() == module::Mode::F8E4M3) {
    attrs.push_back(rewriter.getNamedAttr(
        "out_f8_scales",
        rewriter.getF64ArrayAttr(ArrayRef<double>{*quant_scale_v})));
    auto newType = getQuantF8E4M3Type(op.getOutput());
    auto newValue =
        CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);
    rewriter.replaceOp(op, {newValue});
  } else {
    auto newType = getQuantF8E5M2Type(op.getOutput());
    auto newValue =
        CreateConvOp(rewriter, p.dims, op->getLoc(), newType, operands, attrs);

    rewriter.replaceOp(op, {newValue});
  }
}

void ConvLowering::LoweringQuantized(PatternRewriter &rewriter,
                                     top::ConvOp op) const {
  if (module::isUniformQuantized(op.getInput()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  bool out_i32 = module::isUniformQuantized(op.getOutput()) == false;
  auto p = op.parseParam();
  auto input_qtype = module::getUniformQuantizedType(op.getInput());
  auto filter_type = op.getFilter().getType().cast<RankedTensorType>();
  auto filter_qtype = filter_type.getElementType()
                          .dyn_cast<quant::UniformQuantizedPerAxisType>();
  int32_t filter_zeroPoint;

  if (!filter_qtype) {
    auto filter_qtype = module::getUniformQuantizedType(op.getFilter());
    filter_zeroPoint = filter_qtype.getZeroPoint();
  } else {
    filter_zeroPoint = filter_qtype.getZeroPoints()[0];
  }
  rewriter.setInsertionPointAfter(op);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(op.getFilter());

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool with_bias = true;
  if (input_zeroPoint != 0) {
    // merge input_zeroPoint to bias
    auto filter_stype = module::getStorageType(op.getFilter());
    i32_array_t bias_quant;
    std::shared_ptr<std::vector<int8_t>> filter_quant;
    filter_quant =
        cast<top::WeightOp>(op.getFilter().getDefiningOp()).read<int8_t>();
    if (p.has_bias) {
      bias_quant =
          cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
    } else {
      // bias_quant->resize(p.oc, 0);
      bias_quant = i32_array_t(new std::vector<int32_t>(p.oc, 0));
    }
    int64_t oc = filter_type.getShape()[0];
    int64_t kernel_size = filter_type.getNumElements() / oc;

    if (filter_stype.isUnsignedInteger(8)) {
      for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
        for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
          bias_quant->data()[oc_ind] -=
              input_zeroPoint *
              ((uint8_t)filter_quant->at(kernel_ind + oc_ind * kernel_size) -
               filter_zeroPoint);
        }
      }
    } else {
      for (size_t oc_ind = 0; oc_ind < oc; ++oc_ind) {
        for (size_t kernel_ind = 0; kernel_ind < kernel_size; ++kernel_ind) {
          bias_quant->data()[oc_ind] -=
              input_zeroPoint *
              (filter_quant->at(kernel_ind + oc_ind * kernel_size) -
               filter_zeroPoint);
        }
      }
    }
    auto bias_type = RankedTensorType::get({oc}, rewriter.getI32Type());
    auto new_bias =
        top::WeightOp::create(op, "_merge_bias", *bias_quant, bias_type);
    operands.push_back(new_bias);
  } else if (p.has_bias) {
    auto bias_stype = module::getStorageType(op.getBias());
    auto bias_new_type =
        RankedTensorType::get(module::getShape(op.getBias()), bias_stype);
    op.getBias().setType(bias_new_type);
    operands.push_back(op.getBias());
  } else {
    with_bias = false;
    operands.push_back(op.getBias());
  }
  if (filter_zeroPoint)
    attrs.push_back(rewriter.getNamedAttr(
        "kernel_zp", rewriter.getI64IntegerAttr(filter_zeroPoint)));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));
  if (out_i32) {
    auto newValue = CreateConvOp(rewriter, p.dims, op->getLoc(),
                                 op.getOutput().getType(), operands, attrs);
    rewriter.replaceOp(op, {newValue});
    return;
  }

  auto newType = RankedTensorType::get(module::getShape(op.getOutput()),
                                       rewriter.getI32Type());
  auto new_name = module::getName(op.getOperation()).str() + "_int32";
  auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));

  auto newValue =
      CreateConvOp(rewriter, p.dims, name_loc, newType, operands, attrs);

  // generate requant param
  auto output_qtype = module::getUniformQuantizedType(op.getOutput());
  int quant_size = 1;
  SmallVector<int64_t> shift(1);
  SmallVector<int64_t> multiplier(1);
  auto input_scale = input_qtype.getScale();
  auto output_scale = output_qtype.getScale();

  if (!filter_qtype) {
    auto filter_qtype = module::getUniformQuantizedType(op.getFilter());
    filter_zeroPoint = filter_qtype.getZeroPoint();
    auto filter_scale = filter_qtype.getScale();
    const double effective_output_scale =
        input_scale * filter_scale / output_scale;
    QuantizeMultiplier(effective_output_scale, &multiplier[0], &shift[0]);
  } else {
    auto filter_scales = filter_qtype.getScales();
    filter_zeroPoint = filter_qtype.getZeroPoints()[0];
    quant_size = filter_scales.size();
    shift.resize(quant_size);
    multiplier.resize(quant_size);
    // tensorflow/lite/kernels/kernel_util.cc::PopulateConvolutionQuantizationParams
    // Populate multiplier and shift using affine quantization.
    for (auto filter : llvm::enumerate(filter_scales)) {
      const double effective_output_scale =
          input_scale * filter.value() / output_scale;
      QuantizeMultiplier(effective_output_scale, &multiplier[filter.index()],
                         &shift[filter.index()]);
    }
  }

  // do requant
  if (quant_size == 1) {
    newValue =
        do_requant(op->getLoc(), newValue, op.getOutput().getType(), true,
                   multiplier[0], shift[0], tpu::RequantMode::TFLite_LShift);
  } else {
    std::vector<int32_t> quant;
    std::vector<int64_t> quant_shape(module::getShape(op.getInput()).size(),
                                     1l);
    quant_shape[1] = quant_size;
    if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
        module::isSGTPUV8()) {
      quant.resize(quant_size * 2, 0);
      for (int i = 0; i < quant_size; ++i) {
        quant[i * 2] = multiplier[i];
        quant[i * 2 + 1] =
            (((int32_t)shift[i]) & 0xffff) |
            (((int32_t)output_qtype.getZeroPoint() & 0xffff) << 16);
      }
      quant_shape.back() = 2;
    } else {
      quant.resize(quant_size * 3, 0);
      for (int i = 0; i < quant_size; ++i) {
        quant[i * 3] = multiplier[i];
        quant[i * 3 + 1] = shift[i];
        quant[i * 3 + 2] = output_qtype.getZeroPoint();
      }
      quant_shape.back() = 3;
    }
    auto quant_type = RankedTensorType::get(quant_shape, rewriter.getI32Type());
    auto quantValue = top::WeightOp::create(op, "quant", quant, quant_type);
    newValue =
        do_requant(op->getLoc(), newValue, quantValue, op.getOutput().getType(),
                   true, tpu::RequantMode::TFLite_LShift);
  }
  rewriter.replaceOp(op, {newValue});
}

} // namespace bm1684x
} // namespace tpu_mlir
