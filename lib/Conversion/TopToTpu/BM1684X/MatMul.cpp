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

void MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_f32<tpu::MatMulOp>(rewriter, op, 5);
}

void MatMulLowering::LoweringINT8(PatternRewriter &rewriter, top::MatMulOp op,
                                  bool asymmetric) const {
  // refer quantize_convlike_layer_int8
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto p = op.parseParam();
  std::vector<int64_t> scales;
  std::vector<int64_t> shifts;
  bool fuse_rq = false;
  bool use_perchannel = false;

  bool output_int16 = op->hasAttr("output_int16");
  double i16_out_scale = 0, i16_out_zp = 0;
  auto per_channel_quant_attr = op->getAttr("matmulPerchannelQuant");
  if (per_channel_quant_attr) {
    if (per_channel_quant_attr.isa<mlir::BoolAttr>()) {
      mlir::BoolAttr per_channel_quant =
          per_channel_quant_attr.cast<mlir::BoolAttr>();
      use_perchannel = per_channel_quant.getValue();
    }
  }

  if (p.batch > 1 && p.with_bias != 0) {
    auto bias_size = module::getNumElements(op.getBias());
    if (bias_size > p.N)
      llvm_unreachable("BatchMatMul does not support batch-bias yet.");
  }
  int64_t left_num_dims = module::getShape(op.getInput()).size();
  if (module::isWeight(op.getInput())) {
    if (module::isMARS3() || module::isSGTPUV8())
      LoweringBF16(rewriter, op);
    else
      LoweringF16(rewriter, op);
    return;
  }

  // note: trick for imgToCol pattern
  const auto defByWeightOp = [&](Operation *Op) {
    using TYPE = std::function<std::pair<bool, Operation *>(Operation * op)>;
    TYPE f;
    f = [&](Operation *op) -> decltype(std::declval<TYPE>()(op)) {
      if (isa<top::WeightOp>(op))
        return std::make_pair(true, op);
      else if (!isa<top::ReshapeOp>(op)) {
        return std::make_pair(false, nullptr);
      } else
        return f(op->getOperand(0).getDefiningOp());
    };
    return f(Op);
  };

  const auto defByWeightReshapeOp =
      [&](Operation *op) -> std::pair<bool, Operation *> {
    if (op && isa<top::ReshapeOp>(op) && defByWeightOp(op).first)
      return std::make_pair(true, op);
    else
      return std::make_pair(false, nullptr);
  };

  auto eliminateInvalidOp = [&](bool yes, Operation *op) {
    if (yes && isa<top::ReshapeOp>(op))
      rewriter.eraseOp(op);
    return true;
  };

  auto [righIsReshapeOp, rightReshapeOp] =
      defByWeightReshapeOp(op.getRight().getDefiningOp());
  auto [biasIsReshapeOp, biasReshapeOp] =
      defByWeightReshapeOp(op.getBias().getDefiningOp());
  auto [isWeight, rightOp] = defByWeightOp(op.getRight().getDefiningOp());

  if (isWeight) {
    auto filterOp = dyn_cast<top::WeightOp>(rightOp);
    auto filter_f32 = filterOp.read<float>();
    int64_t in_zp = 0, out_zp = 0;
    double in_scale = 1, out_scale = 1;
    std::vector<double> w_scale(p.N, 1.0);
    bool input_asymmetric = op->hasAttr("input_asym");
    module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp,
                                 input_asymmetric || asymmetric);
    module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
    if (output_int16) {
      out_scale = out_scale * 255 / 65535;
      // out_zp = asymmetric ? (-32768 + (128 + out_zp) * 65535 / 255) : 0;
      out_zp = 0;
      i16_out_scale = out_scale;
      i16_out_zp = out_zp;
    }
    bool right_transpose = op.getRightTranspose();
    if (p.batch > 1 && in_zp != 0) { // Cannot merge zp to bias in BatchMatMul
      LoweringF32(rewriter, op);
      return;
    }
    if (filterOp.getScale().has_value()) {
      auto weight_scale_v = module::getF64Array(filterOp.getScale().value());
      if (use_perchannel) {
        assert(weight_scale_v->size() == p.N);
        w_scale.assign(weight_scale_v->data(),
                       weight_scale_v->data() + weight_scale_v->size());
      } else {
        w_scale[0] = weight_scale_v->data()[0];
      }
    } else {
      if (use_perchannel) {
        if (!right_transpose) {
          for (int n = 0; n < p.N; n++) {
            float tmp = std::abs(filter_f32->data()[n]);
            for (int k = 1; k < p.K; k++) {
              tmp = tmp >= std::abs(filter_f32->data()[n + k * p.N])
                        ? tmp
                        : std::abs(filter_f32->data()[n + k * p.N]);
            }
            w_scale[n] = tmp / 127.0;
          }
        } else {
          for (size_t n = 0; n < p.N; n++) {
            float tmp = std::abs(filter_f32->data()[n * p.K]);
            for (size_t k = 1; k < p.K; k++) {
              tmp = tmp >= std::abs(filter_f32->data()[k + n * p.K])
                        ? tmp
                        : std::abs(filter_f32->data()[k + n * p.K]);
            }
            w_scale[n] = tmp / 127.0;
          }
        }
      } else {
        w_scale[0] = findMaxabs(filter_f32->data(), filter_f32->size()) / 127.0;
      }
    }

    auto filter_int8 =
        std::make_shared<std::vector<int8_t>>(filter_f32->size());
    if (use_perchannel) {
      if (!right_transpose) {
        for (size_t k = 0; k < p.K; k++) {
          for (size_t n = 0; n < p.N; n++) {
            filter_int8->at(k * p.N + n) =
                to_int8(filter_f32->at(k * p.N + n) / w_scale[n]);
          }
        }
      } else {
        for (size_t n = 0; n < p.N; n++) {
          for (size_t k = 0; k < p.K; k++) {
            filter_int8->at(n * p.K + k) =
                to_int8(filter_f32->at(n * p.K + k) / w_scale[n]);
          }
        }
      }
    } else {
      for (uint64_t t = 0; t < filter_f32->size(); t++) {
        filter_int8->at(t) = to_int8(filter_f32->at(t) / w_scale[0]);
      }
    }

    i32_array_t bias_int32;
    std::shared_ptr<std::vector<float>> bias_fp32;
    if (p.with_bias) {
      auto [isWeight, bias] = defByWeightOp(op.getBias().getDefiningOp());
      assert(isWeight);
      auto biasOp = cast<top::WeightOp>(bias);
      bias_fp32 = biasOp.read<float>();
      bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
    } else if (in_zp) {
      bias_int32 = std::make_shared<std::vector<int32_t>>(p.N);
    }

    for (int j = 0; j < p.N; j++) { // vector [1xN]
      int64_t bias_w_xz = 0;
      for (int i = 0; i < p.K; i++) {
        bias_w_xz += (int64_t)filter_int8->at(i * p.N + j) * in_zp;
      }

      if (p.with_bias) {
        if (use_perchannel) {
          bias_int32->data()[j] = std::round(
              bias_fp32->at(j) / (w_scale[j] * in_scale) - bias_w_xz);
        } else {
          bias_int32->data()[j] = std::round(
              bias_fp32->at(j) / (w_scale[0] * in_scale) - bias_w_xz);
        }
      } else if (in_zp) {
        bias_int32->data()[j] = -bias_w_xz;
      }
    }
    bool with_bias = p.with_bias || in_zp != 0;
    std::vector<float> scale_f;
    if (use_perchannel) {
      int common_shift = -1;
      for (size_t n = 0; n < p.N; n++) {
        scale_f.push_back(in_scale * w_scale[n] / out_scale);
      }
      for (int shift_limit = 32; shift_limit > 0; shift_limit--) {
        int max_shift = 0;
        for (size_t n = 0; n < p.N; n++) {
          int scale_ = 1, shift_ = 0;
          get_scale_and_shift(scale_f[n], scale_, shift_, shift_limit);
          scales.push_back(scale_);
          shifts.push_back(shift_);
          max_shift = shift_ > max_shift ? shift_ : max_shift;
        }
        bool overflow = false;
        for (size_t n = 0; n < p.N; n++) {
          long long multi = ((long long)scales[n]) << (max_shift - shifts[n]);
          if (multi > 0x80000000ll) {
            overflow = true;
            scales[n] = 0x7fffffff;
            shifts[n] = max_shift;
          } else {
            scales[n] = scales[n] << (max_shift - shifts[n]);
            shifts[n] = max_shift;
          }
        }
        if (!overflow) {
          common_shift = max_shift;
          break;
        }
        scales.clear();
        shifts.clear();
      }
      if (common_shift < 0) {
        llvm_unreachable("found overflow in per-channel quant");
        ;
      }
    } else {
      int scale_ = 1, shift_ = 0;
      scale_f.push_back(in_scale * w_scale[0] / out_scale);
      get_scale_and_shift(scale_f[0], scale_, shift_);
      shifts.push_back(shift_);
      scales.push_back(scale_);
    }
    auto filter_type = op.getRight().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
    auto new_filter =
        top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
    operands.push_back(op.getInput());
    operands.push_back(new_filter);
    auto new_bias = op.getBias();
    if (with_bias) {
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = p.N;
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      new_bias = top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = module::getNoneOp(op);
      operands.push_back(none);
    }
    if (use_perchannel) {
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = p.N;
      std::vector<int> scales32;
      for (auto s : scales)
        scales32.push_back((int)s);
      new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      auto multi = top::WeightOp::create(op, "multi_int32", scales32, new_type);
      operands.push_back(multi);
      fuse_rq = true;
    }
  } else if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  } else { // mutable tensor or MatMul
    int64_t in_zp = 0, w_zp = 0, out_zp = 0;
    double in_scale = 1, w_scale = 1, out_scale = 1;
    int scale = 1, shift = 0;

    module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
    module::getScaleAndZeroPoint(op.getRight(), w_scale, w_zp, asymmetric);
    module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
    if (output_int16) {
      out_scale = out_scale * 255 / 65535;
      // out_zp = asymmetric ? (-32768 + (128 + out_zp) * 65535 / 255) : 0;
      out_zp = 0;
      i16_out_scale = out_scale;
      i16_out_zp = out_zp;
    }
    float scale_f = in_scale * w_scale / out_scale;
    get_scale_and_shift(scale_f, scale, shift, 32);
    shifts.push_back(shift);
    scales.push_back(scale);
    for (auto operand : op.getOperands())
      operands.push_back(operand);
    if (p.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      int bias_n = bias_fp32->size();
      auto bias_int32 = std::make_shared<std::vector<int32_t>>(bias_n);
      for (int j = 0; j < bias_n; j++) {
        bias_int32->data()[j] =
            std::round(bias_fp32->at(j) / (w_scale * in_scale));
      }
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = bias_n;
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      auto new_bias =
          top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands[2] = new_bias;
    }
  }
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(shifts)));
  attrs.push_back(
      rewriter.getNamedAttr("multipliers", rewriter.getI64ArrayAttr(scales)));

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (!fuse_rq) {
    auto noneOp_multi = module::getNoneOp(op);
    operands.push_back(noneOp_multi);
  }
  // buffer
  operands.push_back(module::getNoneOp(op));
  Type newType;
  if (output_int16) {
    newType = getQuantIntType(op.getOutput(), i16_out_scale, i16_out_zp, 16);
  } else {
    newType = getQuantInt8Type(op.getOutput(), asymmetric);
  }
  auto newmm =
      rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
  if (fuse_rq) {
    newmm.setFuseRqAttr(rewriter.getBoolAttr(true));
  }
  // trick for Img2Col
  eliminateInvalidOp(righIsReshapeOp, rightReshapeOp);
  eliminateInvalidOp(biasIsReshapeOp, biasReshapeOp);
}

void MatMulLowering::LoweringINT4(PatternRewriter &rewriter, top::MatMulOp op,
                                  bool asymmetric) const {

  // refer quantize_convlike_layer_int8
  llvm::errs() << "start MatMul LoweringINT4, call LoweringINT8, name:"
               << module::getName(op.getOperation()).str() << "\n";
  // return LoweringINT8(rewriter, op, asymmetric);

  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto p = op.parseParam();
  int scale = 1, shift = 0;
  if (p.batch > 1 && p.with_bias != 0) {
    auto bias_size = module::getNumElements(op.getBias());
    if (bias_size > p.N)
      llvm_unreachable("BatchMatMul does not support batch-bias yet.");
  }
  double in_int8_scale;
  int64_t in_int8_zp;
  bool all_next_layer_is_int8 = true;
  bool all_next_layer_is_int4 = true;
  double out_int8_scale =
      op.getOutInt8Scale().value_or(APFloat(1.0)).convertToDouble();
  double out_int8_zp =
      op.getOutInt8Zp().value_or(APFloat(0.0)).convertToDouble();
  int64_t in_zp = 0, out_zp = 0;
  double in_scale = 1, out_scale = 1, w_scale = 1;

  int64_t left_num_dims = module::getShape(op.getInput()).size();
  if (auto filterOp = dyn_cast<top::WeightOp>(op.getRight().getDefiningOp())) {
    auto filter_f32 = filterOp.read<float>();
    int bitwidth = 4;
    Value value;
    if (op.getInInt4Scale().has_value()) {
      // bool find = false;
      // for (auto user : op.getInput().getDefiningOp()->getUsers()) {
      //   if (isa<tpu::RequantFpOp>(user)) {
      //     find = true;
      //     operands.push_back(user->getResults()[0]);
      //     break;
      //   }
      // }
      // if (!find) {
      // 存在int4的输入scale，说明上一层是int8，故输入tensor也是int8，需要requant为int4
      in_scale =
          op->getAttr("in_int4_scale").cast<FloatAttr>().getValueAsDouble();
      in_zp = op->getAttr("in_int4_zp").cast<FloatAttr>().getValueAsDouble();
      module::getScaleAndZeroPoint(op.getInput(), in_int8_scale, in_int8_zp,
                                   asymmetric);
      auto output_type = getQuantIntType(op.getInput(), in_scale, in_zp, 4);
      double int4_scale = in_int8_scale / in_scale; // 将int8转为int4的rq参数
      double offset = in_zp - in_int8_zp * int4_scale;
      auto to_name = "to_b4_for_" + module::getName(op.getOperation()).str();
      value =
          do_requantFp(op.getInput(), int4_scale, offset, output_type, to_name);
      operands.push_back(value);
      // }
    } else { // 输入tensor也是int4
      operands.push_back(op.getInput());
      module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric,
                                   bitwidth);
    }
    module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric,
                                 bitwidth);
    if (p.batch > 1 && in_zp != 0) { // Cannot merge zp to bias in BatchMatMul
      LoweringF32(rewriter, op);
      return;
    }
    if (filterOp.getScale().has_value()) {
      auto weight_scale_v = module::getF64Array(filterOp.getScale().value());
      w_scale = weight_scale_v->data()[0];
    } else {
      double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
      w_scale = w_max / 7.0;
    }

    auto filter_int8 =
        std::make_shared<std::vector<int8_t>>(filter_f32->size());
    for (uint64_t t = 0; t < filter_f32->size(); t++) {
      filter_int8->at(t) = to_int8(filter_f32->at(t) / w_scale);
    }

    i32_array_t bias_int32;
    std::shared_ptr<std::vector<float>> bias_fp32;
    if (p.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      bias_fp32 = biasOp.read<float>();
      bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
    } else if (in_zp) {
      bias_int32 = std::make_shared<std::vector<int32_t>>(p.N);
    }

    for (int j = 0; j < p.N; j++) { // vector [1xN]
      int64_t bias_w_xz = 0;
      for (int i = 0; i < p.K; i++) {
        bias_w_xz += (int64_t)filter_int8->at(i * p.N + j) * in_zp;
      }

      if (p.with_bias) {
        bias_int32->data()[j] =
            std::round(bias_fp32->at(j) / (w_scale * in_scale) - bias_w_xz);
      } else if (in_zp) {
        bias_int32->data()[j] = -bias_w_xz;
      }
    }

    for (auto user : op->getUsers()) {
      if (module::isInt4Op(user)) {
        all_next_layer_is_int8 = false;
      } else {
        all_next_layer_is_int4 = false;
      }

      if (isa<ReturnOp>(user)) {
        all_next_layer_is_int4 = true;
        all_next_layer_is_int8 = false;
        break;
      }
    }

    llvm::errs() << "all_next_layer_is_int4:" << all_next_layer_is_int4
                 << ",all_next_layer_is_int8:" << all_next_layer_is_int8
                 << "\n";
    if (all_next_layer_is_int8)
      llvm::errs() << "directly output int8\n";
    else
      llvm::errs() << "directly output int4\n";

    bool with_bias = p.with_bias || in_zp != 0;
    float scale_f;
    if (all_next_layer_is_int8) {
      scale_f = w_scale * in_scale / out_int8_scale;
    } else {
      scale_f = in_scale * w_scale / out_scale;
    }
    get_scale_and_shift(scale_f, scale, shift, 32);
    auto filter_type = op.getRight().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
    auto new_filter =
        top::WeightOp::create(op, "filter_int4", *filter_int8, new_type);
    // operands.push_back(op.getInput());
    operands.push_back(new_filter);
    auto new_bias = op.getBias();
    if (with_bias) {
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = p.N;
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      new_bias = top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = module::getNoneOp(op);
      operands.push_back(none);
    }
  } else if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  } else { // mutable tensor or MatMul
    int64_t in_zp = 0, w_zp = 0, out_zp = 0;
    double in_scale = 1, w_scale = 1, out_scale = 1;
    module::getScaleAndZeroPoint(op.getInput(), in_scale, in_zp, asymmetric);
    module::getScaleAndZeroPoint(op.getRight(), w_scale, w_zp, asymmetric);
    module::getScaleAndZeroPoint(op.getOutput(), out_scale, out_zp, asymmetric);
    float scale_f = in_scale * w_scale / out_scale;
    get_scale_and_shift(scale_f, scale, shift, 32);
    for (auto operand : op.getOperands())
      operands.push_back(operand);
    if (p.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      int bias_n = bias_fp32->size();
      auto bias_int32 = std::make_shared<std::vector<int32_t>>(bias_n);
      for (int j = 0; j < bias_n; j++) {
        bias_int32->data()[j] =
            std::round(bias_fp32->at(j) / (w_scale * in_scale));
      }
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = bias_n;
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      auto new_bias =
          top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands[2] = new_bias;
    }
  }

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }

  if (!all_next_layer_is_int8 && !all_next_layer_is_int4) {
    // to int32, and then requant to int8
    auto convType = RankedTensorType::get(module::getShape(op.getOutput()),
                                          rewriter.getI32Type());
    auto matmul_int32_name =
        module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(matmul_int32_name));
    auto noneOp_multi = module::getNoneOp(op);
    operands.push_back(noneOp_multi);
    // buffer
    operands.push_back(module::getNoneOp(op));
    auto matmul_int32_out =
        rewriter.create<tpu::MatMulOp>(name_loc, convType, operands, attrs);

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
      // requant
      std::vector<int32_t> quant;
      int64_t quant_w_size = 0;
      if (module::isBM1688() || module::isSG2380() || module::isMARS3() ||
          module::isSGTPUV8()) {
        quant_w_size = 2;
        quant.resize(quant_w_size, 0);
        quant[i * 2] = scale;
        quant[i * 2 + 1] =
            ((-(int32_t)shift) & 0xffff) | (((int32_t)out_zp & 0xffff) << 16);
      } else {
        quant_w_size = 3;
        quant.resize(quant_w_size, 0);
        quant[i * 3] = scale;
        quant[i * 3 + 1] = -shift;
        quant[i * 3 + 2] = out_zp;
      }
      auto quant_type = RankedTensorType::get({1, 1, 1, 1, quant_w_size},
                                              rewriter.getI32Type());
      auto quant_value = top::WeightOp::create(op, w_name, quant, quant_type);

      auto newValue =
          do_requant(requant_name_loc, matmul_int32_out, quant_value, newType,
                     true, tpu::RequantMode::MultiplierShift);

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
    rewriter.replaceOp(op, matmul_int32_out);
  } else {
    attrs.push_back(
        rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(shift)));
    attrs.push_back(
        rewriter.getNamedAttr("multipliers", rewriter.getI64ArrayAttr(scale)));

    auto newType = getQuantInt4Type(op.getOutput(), asymmetric);
    if (all_next_layer_is_int8) {
      newType = getQuantIntType(op.getOutput(), out_int8_scale, out_int8_zp);
    }
    auto noneOp_multi = module::getNoneOp(op);
    operands.push_back(noneOp_multi);
    // buffer
    operands.push_back(module::getNoneOp(op));
    auto newOp =
        rewriter.create<tpu::MatMulOp>(op->getLoc(), newType, operands, attrs);
    rewriter.replaceOp(op, {newOp.getOutput()});
  }
}
void MatMulLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::MatMulOp op) const {
  bool bias_use_fp32 = !(module::isBM1684X() || module::isBM1690Family());
  auto newType = getQuantBF16Type(op->getResult(0));
  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);

    if (auto wOp = dyn_cast<top::WeightOp>(in.getDefiningOp())) {
      // only linear layer will be replaced
      if (i == 1 && op.getWeightBits().has_value() &&
          wOp.getType().cast<RankedTensorType>().getShape().size() == 2) {
        auto noneOp = module::getNoneOp(op);
        operands.insert(operands.end(),
                        {in, noneOp, noneOp, op->getOperand(2)});
        std::vector<NamedAttribute> attrs;
        auto weight_bits =
            rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
        attrs.push_back(weight_bits);
        rewriter.replaceOpWithNewOp<tpu::A16MatMulOp>(op, newType, operands,
                                                      attrs);
        return;
      }
      if (i == 2 && bias_use_fp32) {
        operands.push_back(in);
      } else {
        operands.push_back(wOp.clone_bf16(op));
      }
    } else {
      operands.push_back(in);
    }
  }
  auto noneOp_multi = module::getNoneOp(op);
  operands.push_back(noneOp_multi);
  // buffer
  operands.push_back(module::getNoneOp(op));
  auto newOp = rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands,
                                                          op->getAttrs());
  if (!module::isNone(operands[2]) && !bias_use_fp32 &&
      module::isBM1690Family()) {
    auto biasOp = dyn_cast<top::WeightOp>(newOp.getOperand(2).getDefiningOp());
    auto bf16_bias = biasOp.clone_bf16(newOp);
    newOp.setOperand(2, bf16_bias);
  }
}

void MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  bool bias_use_fp32 = !(module::isBM1684X() || module::isBM1690Family());
  auto newType = getQuantF16Type(op->getResult(0));
  std::vector<Value> operands;
  for (int i = 0; i < op->getNumOperands(); ++i) {
    auto in = op->getOperand(i);

    if (auto wOp = dyn_cast<top::WeightOp>(in.getDefiningOp())) {
      // only linear layer will be replaced
      if (i == 1 && op.getWeightBits().has_value() &&
          wOp.getType().cast<RankedTensorType>().getShape().size() == 2) {
        auto noneOp = module::getNoneOp(op);
        operands.insert(operands.end(),
                        {in, noneOp, noneOp, op->getOperand(2)});
        std::vector<NamedAttribute> attrs;
        auto weight_bits =
            rewriter.getNamedAttr("weight_bits", op.getWeightBitsAttr());
        attrs.push_back(weight_bits);
        if (true == op.getDoRelu()) {
          auto name = module::getName(op->getResult(0));
          auto matmul_loc =
              NameLoc::get(rewriter.getStringAttr(name.str() + "_a16matmul"));
          auto a16matmul_op = rewriter.create<tpu::A16MatMulOp>(
              matmul_loc, newType, operands, attrs);
          std::vector<NamedAttribute> relu_attrs;
          auto relu_limit =
              rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr());
          relu_attrs.push_back(relu_limit);
          rewriter.replaceOpWithNewOp<tpu::ReluOp>(
              op, newType, ValueRange{a16matmul_op.getOutput()}, relu_attrs);
          return;
        }
        rewriter.replaceOpWithNewOp<tpu::A16MatMulOp>(op, newType, operands,
                                                      attrs);
        return;
      }
      if (i == 2 && bias_use_fp32) {
        ASSERT_OP(module::getStorageType(in).isF32() && "bias has to be f32",
                  op);
        operands.push_back(in);
      } else {
        operands.push_back(wOp.clone_f16(op));
      }
    } else {
      operands.push_back(in);
    }
  }
  auto noneOp_multi = module::getNoneOp(op);
  operands.push_back(noneOp_multi);
  // buffer
  operands.push_back(module::getNoneOp(op));
  auto newOp = rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands,
                                                          op->getAttrs());
  if (!module::isNone(operands[2]) && !bias_use_fp32 &&
      module::isBM1690Family()) {
    auto biasOp = dyn_cast<top::WeightOp>(newOp.getOperand(2).getDefiningOp());
    auto f16_bias = biasOp.clone_f16(newOp);
    newOp.setOperand(2, f16_bias);
  }
}

void MatMulLowering::LoweringF8(PatternRewriter &rewriter,
                                top::MatMulOp op) const {
  // Y = W*x + B => ScaleY * Yf8 =ScaleW * Wf8 * ScaleX * Xf8 + Bf32
  // Yf8 = scaleW * scaleX / scaleY * Wf8 * Xf8 + 1/ScaleY * Bf32
  // Bf32 / (scaleX * scaleW) * ((scaleX * scaleW) / scaleY)

  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  auto p = op.parseParam();
  auto in = op.getInput();
  auto out = op.getOutput();
  double in_scale = 1.0, out_scale = 1.0;

  if (module::getMode() == module::Mode::F8E5M2) {
    operands.push_back(op.getInput());
    if (auto weight = dyn_cast<top::WeightOp>(op.getRight().getDefiningOp())) {
      auto new_w = weight.clone_f8e5m2(op);
      operands.push_back(new_w);
    } else {
      operands.push_back(op.getRight());
    }
    if (p.with_bias) {
      operands.push_back(op.getBias());
    } else {
      operands.push_back(module::getNoneOp(op));
    }
    auto noneOp_multi = module::getNoneOp(op);
    operands.push_back(noneOp_multi);
    // buffer
    operands.push_back(module::getNoneOp(op));
    auto newType = getQuantF8E5M2Type(op.getOutput());
    rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
    return;
  }
  auto qtype_in = module::getCalibratedType(in);
  auto qtype_out = module::getCalibratedType(out);
  auto out_max = qtype_out.getMax();
  in_scale = qtype_in.getMax() / get_f8e4m3_max();
  out_scale = out_max / get_f8e4m3_max();

  if (p.batch > 1 && p.with_bias != 0) {
    auto bias_size = module::getNumElements(op.getBias());
    if (bias_size > p.N)
      llvm_unreachable("BatchMatMul does not support batch-bias yet.");
  }
  int64_t left_num_dims = module::getShape(op.getInput()).size();
  Value newWeight;
  std::vector<double> sclae_f_v;
  if (auto weightOp = dyn_cast<top::WeightOp>(op.getRight().getDefiningOp())) {
    newWeight =
        weightOp.clone_f8e4m3(op, module::isSG2262(), op.getRightTranspose());
    auto w_op = dyn_cast<top::WeightOp>(newWeight.getDefiningOp());
    f64_array_t weight_scale_v = module::getF64Array(w_op.getScale().value());
    operands.push_back(in);
    operands.push_back(newWeight);
    if (module::isSG2262()) {
      for (int j = 0; j < p.N; j++)
        sclae_f_v.push_back(in_scale * weight_scale_v.get()->at(j) / out_scale);
    } else {
      sclae_f_v.push_back(in_scale * weight_scale_v.get()->at(0) / out_scale);
    }
    if (p.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto b_value = biasOp.read<float>();
      for (int j = 0; j < p.N; j++) {
        if (module::isSG2262()) {
          b_value->at(j) =
              b_value->at(j) / (in_scale * weight_scale_v.get()->at(j));
        } else {
          b_value->at(j) =
              b_value->at(j) / (in_scale * weight_scale_v.get()->at(0));
        }
      }
      auto new_bias = op.getBias();
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = p.N;
      auto new_type = RankedTensorType::get(shape, rewriter.getF32Type());
      new_bias = top::WeightOp::create(op, "bias_f32", *b_value, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = module::getNoneOp(op);
      operands.push_back(none);
    }
  } else {
    auto right = op.getRight();
    auto qtype_right = module::getCalibratedType(right);
    double w_scale = qtype_right.getMax() / get_f8e4m3_max();
    sclae_f_v.push_back(in_scale * w_scale / out_scale);
    for (auto operand : op.getOperands())
      operands.push_back(operand);
    if (p.with_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto b_value = biasOp.read<float>();
      int bias_n = b_value->size();
      for (int j = 0; j < bias_n; j++)
        b_value->at(j) = b_value->at(j) / (in_scale * w_scale);
      auto new_bias = op.getBias();
      std::vector<int64_t> shape(left_num_dims, 1);
      shape[left_num_dims - 1] = p.N;
      auto new_type = RankedTensorType::get(shape, rewriter.getF32Type());
      new_bias = top::WeightOp::create(op, "bias_f32", *b_value, new_type);
      operands[2] = new_bias;
    }
  }
  bool with_bias = !module::isNone(op.getBias());
  attrs.push_back(rewriter.getNamedAttr(
      "out_f8_scales", rewriter.getF64ArrayAttr(ArrayRef<double>{sclae_f_v})));
  attrs.push_back(rewriter.getNamedAttr(
      "quant_mode", tpu::RequantModeAttr::get(op->getContext(),
                                              tpu::RequantMode::OnlyScale)));
  attrs.push_back(
      rewriter.getNamedAttr("with_bias", rewriter.getBoolAttr(with_bias)));

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto noneOp_multi = module::getNoneOp(op);
  operands.push_back(noneOp_multi);
  // buffer
  operands.push_back(module::getNoneOp(op));
  auto newType = getQuantF8E4M3Type(op.getOutput());
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void MatMulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::MatMulOp op) const {
  if (!module::isUniformQuantized(op.getInput(), op.getRight())) {
    UNREACHABLE_OP("input output should be quantized", op);
  }
  bool out_i32 = module::isUniformQuantized(op.getOutput()) == false;
  auto p = op.parseParam();
  // assert(batch == 1);
  auto input_qtype = module::getUniformQuantizedType(op.getInput());
  auto right_qtype = module::getUniformQuantizedType(op.getRight());
  int64_t left_num_dims = module::getShape(op.getInput()).size();
  int32_t right_zero_point = right_qtype.getZeroPoint();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  if (module::isWeight(op.getRight())) {
    auto right_stype = module::getStorageType(op.getRight());
    auto right_new_type =
        RankedTensorType::get(module::getShape(op.getRight()), right_stype);
    op.getRight().setType(right_new_type);
  }

  operands.push_back(op.getRight());
  if (p.with_bias) {
    auto bias_stype = module::getStorageType(op.getBias());
    auto bias_new_type =
        RankedTensorType::get(module::getShape(op.getBias()), bias_stype);
    op.getBias().setType(bias_new_type);
  }

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (right_zero_point)
    attrs.push_back(rewriter.getNamedAttr(
        "right_zp", rewriter.getI64IntegerAttr(right_zero_point)));

  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool can_merge_izp = input_zeroPoint == 0 || module::isWeight(op.getRight());
  int K_idx = op.getRightTranspose() ? 1 : 0;
  int N_idx = op.getRightTranspose() ? 0 : 1;
  if (p.batch > 1) {
    K_idx++;
    N_idx++;
  }
  int64_t row_size = p.K;
  int64_t col_size = p.N;
  i32_array_t bias_quant;
  if (module::isWeight(op.getBias())) {
    bias_quant =
        cast<top::WeightOp>(op.getBias().getDefiningOp()).read<int32_t>();
  } else {
    bias_quant = i32_array_t(new std::vector<int32_t>(col_size, 0));
  }
  std::vector<int64_t> shape(left_num_dims, 1);
  shape[left_num_dims - 1] = col_size;
  auto bias_type = RankedTensorType::get(shape, rewriter.getI32Type());

  Value matValue;
  if (can_merge_izp) {
    if (input_zeroPoint) {
      // merge input_zeroPoint to bias
      std::shared_ptr<std::vector<int8_t>> right_quant;
      right_quant =
          cast<top::WeightOp>(op.getRight().getDefiningOp()).read<int8_t>();
      for (size_t r_ind = 0; r_ind < row_size; ++r_ind) {
        for (size_t c_ind = 0; c_ind < col_size; ++c_ind) {
          auto right_data = op.getRightTranspose()
                                ? right_quant->at(c_ind * row_size + r_ind)
                                : right_quant->at(c_ind + r_ind * col_size);
          bias_quant->data()[c_ind] -=
              input_zeroPoint * (right_data - right_zero_point);
        }
      }
      auto new_bias = top::WeightOp::create(op, "MergedInputZeroPoint",
                                            *bias_quant, bias_type);
      operands.push_back(new_bias);
    } else {
      operands.push_back(op.getBias());
    }
    auto matmul_type = RankedTensorType::get(module::getShape(op.getOutput()),
                                             rewriter.getI32Type());
    auto new_name = module::getName(op.getOperation()).str() + "_no_izp";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    operands.push_back(none);
    // buffer
    operands.push_back(module::getNoneOp(op));
    auto newOp =
        rewriter.create<tpu::MatMulOp>(name_loc, matmul_type, operands, attrs);
    matValue = newOp.getOutput();
  } else {
    // (M * K) (K * N)
    // (Input - izp) Matmul (Right - kzp) ==> (Input) Matmul (Right - kzp) -
    // (izp) Matmul (Right - kzp)
    // - (izp) Matmul (Right - kzp) ==> izp * kzp * K - izp *
    // reduce_sum(Right.col) for each row

    operands.push_back(op.getBias());
    if (input_zeroPoint)
      attrs.push_back(rewriter.getNamedAttr(
          "input_zp", rewriter.getI64IntegerAttr(input_zeroPoint)));
    auto matmul_type = RankedTensorType::get(module::getShape(op.getOutput()),
                                             rewriter.getI32Type());
    auto new_name = module::getName(op.getOperation()).str() + "_int32";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfter(op);
    auto none = module::getNoneOp(op);
    operands.push_back(none);
    // buffer
    operands.push_back(module::getNoneOp(op));
    auto newOp =
        rewriter.create<tpu::MatMulOp>(name_loc, matmul_type, operands, attrs);
    matValue = newOp.getOutput();
  }
  if (out_i32) {
    rewriter.replaceOp(op, {matValue});
    return;
  }

  // generate quantize param
  auto output_qtype = module::getUniformQuantizedType(op.getOutput());
  const double real_multiplier =
      input_qtype.getScale() * right_qtype.getScale() / output_qtype.getScale();
  int64_t multiplier, shift;
  QuantizeMultiplier(real_multiplier, &multiplier, &shift);
  // do requant
  auto newValue =
      do_requant(op->getLoc(), matValue, op.getOutput().getType(), true,
                 multiplier, shift, tpu::RequantMode::TFLite_LShift);
  rewriter.replaceOp(op, {newValue});
}

} // namespace bm1684x
} // namespace tpu_mlir
