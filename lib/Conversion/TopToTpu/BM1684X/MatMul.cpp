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

void MatMulLowering::LoweringF32(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_f32<tpu::MatMulOp>(rewriter, op);
}

void MatMulLowering::LoweringINT8(PatternRewriter &rewriter, top::MatMulOp op,
                                  bool asymmetric) const {
  // refer quantize_convlike_layer_int8
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  int64_t batch, M, K, N;
  bool relu, with_bias, transpose;
  double relu_limit;
  op.parseParam(batch, M, K, N, with_bias, relu, relu_limit, transpose);
  int scale = 1, shift = 0;
  if (batch > 1 && with_bias != 0) {
    auto bias_size = Module::getNumElements(op.bias());
    if (bias_size > N)
      llvm_unreachable("BatchMatMul does not support batch-bias yet.");
  }
  if (auto filterOp = dyn_cast<top::WeightOp>(op.right().getDefiningOp())) {
    auto filter_f32 = filterOp.read<float>();
    int64_t in_zp = 0, out_zp = 0;
    double in_scale = 1, out_scale = 1, w_scale = 1;
    Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);
    Quant::getScaleAndZeroPoint(op.output(), out_scale, out_zp, asymmetric);
    if (batch > 1 && in_zp != 0) { // Cannot merge zp to bias in BatchMatMul
      LoweringF32(rewriter, op);
      return;
    }
    std::shared_ptr<std::vector<double>> weight_scale_v;
    if (filterOp.scale().has_value() && weight_scale_v->size()) {
      weight_scale_v = Module::getF64Array(filterOp.scale().value());
      w_scale = weight_scale_v->data()[0];
    } else {
      double w_max = findMaxabs(filter_f32->data(), filter_f32->size());
      w_scale = w_max / 127.0;
    }

    auto filter_int8 =
        std::make_shared<std::vector<int8_t>>(filter_f32->size());
    for (uint64_t t = 0; t < filter_f32->size(); t++) {
      filter_int8->at(t) = Quant::to_int8(filter_f32->at(t) / w_scale);
    }

    std::shared_ptr<std::vector<int32_t>> bias_int32;
    std::shared_ptr<std::vector<float>> bias_fp32;
    if (with_bias) {
      auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
      bias_fp32 = biasOp.read<float>();
      bias_int32 = std::make_shared<std::vector<int32_t>>(bias_fp32->size());
    } else if (in_zp) {
      bias_int32 = std::make_shared<std::vector<int32_t>>(N);
    }

    for (int j = 0; j < N; j++) { // vector [1xN]
      int64_t bias_w_xz = 0;
      for (int i = 0; i < K; i++) {
        bias_w_xz += (int64_t)filter_int8->at(i * N + j) * in_zp;
      }

      if (with_bias) {
        bias_int32->data()[j] =
            std::round(bias_fp32->at(j) / (w_scale * in_scale) - bias_w_xz);
      } else if (in_zp) {
        bias_int32->data()[j] = -bias_w_xz;
      }
    }
    with_bias = with_bias || in_zp != 0;
    float scale_f = in_scale * w_scale / out_scale;
    get_scale_and_shift(scale_f, scale, shift, 32);
    auto filter_type = op.right().getType().cast<RankedTensorType>();
    auto new_type =
        RankedTensorType::get(filter_type.getShape(), rewriter.getI8Type());
    auto new_filter =
        top::WeightOp::create(op, "filter_int8", *filter_int8, new_type);
    operands.push_back(op.input());
    operands.push_back(new_filter);
    auto new_bias = op.bias();
    if (with_bias) {
      std::vector<int64_t> shape = {N};
      auto new_type = RankedTensorType::get(shape, rewriter.getI32Type());
      new_bias = top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands.push_back(new_bias);
    } else {
      auto none = Module::getNoneOp(op);
      operands.push_back(none);
    }
  } else if (asymmetric) {
    LoweringF32(rewriter, op);
    return;
  } else { // mutable tensor or MatMul
    int64_t in_zp = 0, w_zp = 0, out_zp = 0;
    double in_scale = 1, w_scale = 1, out_scale = 1;
    Quant::getScaleAndZeroPoint(op.input(), in_scale, in_zp, asymmetric);
    Quant::getScaleAndZeroPoint(op.right(), w_scale, w_zp, asymmetric);
    Quant::getScaleAndZeroPoint(op.output(), out_scale, out_zp, asymmetric);
    float scale_f = in_scale * w_scale / out_scale;
    get_scale_and_shift(scale_f, scale, shift, 32);
    for (auto operand : op.getOperands())
      operands.push_back(operand);
    if (with_bias) {
      auto biasOp = cast<top::WeightOp>(op.bias().getDefiningOp());
      auto bias_fp32 = biasOp.read<float>();
      int bias_n = bias_fp32->size();
      auto bias_int32 = std::make_shared<std::vector<int32_t>>(bias_n);
      for (int j = 0; j < bias_n; j++) {
        bias_int32->data()[j] =
            std::round(bias_fp32->at(j) / (w_scale * in_scale));
      }
      auto new_type = RankedTensorType::get({bias_n}, rewriter.getI32Type());
      auto new_bias =
          top::WeightOp::create(op, "bias_int32", *bias_int32, new_type);
      operands[2] = new_bias;
    }
  }
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(shift)));
  attrs.push_back(
      rewriter.getNamedAttr("multipliers", rewriter.getI64ArrayAttr(scale)));

  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  auto newType = getQuantInt8Type(op.output(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, newType, operands, attrs);
}

void MatMulLowering::LoweringBF16(PatternRewriter &rewriter,
                                  top::MatMulOp op) const {
  lowering_common_bf16<tpu::MatMulOp>(rewriter, op);
}

void MatMulLowering::LoweringF16(PatternRewriter &rewriter,
                                 top::MatMulOp op) const {
  lowering_common_f16<tpu::MatMulOp>(rewriter, op);
}

void MatMulLowering::LoweringQuantized(PatternRewriter &rewriter,
                                       top::MatMulOp op) const {
  if (!Quant::isUniformQuantized(op.input(), op.right(), op.output())) {
    llvm_unreachable("input output should be quantized");
  }
  int64_t batch, M, K, N;
  bool relu, with_bias, transpose;
  double relu_limit;
  op.parseParam(batch, M, K, N, with_bias, relu, relu_limit, transpose);
  // assert(batch == 1);
  auto input_qtype = Quant::getUniformQuantizedType(op.input());
  auto right_qtype = Quant::getUniformQuantizedType(op.right());
  auto output_qtype = Quant::getUniformQuantizedType(op.output());

  const double real_multiplier =
      input_qtype.getScale() * right_qtype.getScale() / output_qtype.getScale();
  int64_t multiplier, shift;
  QuantizeMultiplier(real_multiplier, &multiplier, &shift);
  int32_t right_zero_point = right_qtype.getZeroPoint();
  auto ctx = getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands;
  operands.push_back(op.input());
  if (isa<top::WeightOp>(op.right().getDefiningOp())) {
    auto right_stype = Module::getStorageType(op.right());
    auto right_new_type =
        RankedTensorType::get(Module::getShape(op.right()), right_stype);
    op.right().setType(right_new_type);
  }

  operands.push_back(op.right());
  if (with_bias) {
    auto bias_stype = Module::getStorageType(op.bias());
    auto bias_new_type =
        RankedTensorType::get(Module::getShape(op.bias()), bias_stype);
    op.bias().setType(bias_new_type);
  }

  // std::string suffix = "_matmul";
  // std::string new_name = Module::getName(op).str() + suffix;
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    attrs.push_back(attr);
  }
  if (right_zero_point)
    attrs.push_back(rewriter.getNamedAttr(
        "right_zp", rewriter.getI64IntegerAttr(right_zero_point)));

  int32_t input_zeroPoint = input_qtype.getZeroPoint();
  bool can_merge_izp =
      input_zeroPoint == 0 || isa<top::WeightOp>(op.right().getDefiningOp());
  auto right_type = op.right().getType().cast<RankedTensorType>();
  int K_idx = op.right_transpose() ? 1 : 0;
  int N_idx = op.right_transpose() ? 0 : 1;
  if (batch > 1) {
    K_idx++;
    N_idx++;
  }
  int64_t row_size = K;
  int64_t col_size = N;
  std::shared_ptr<std::vector<int32_t>> bias_quant;
  if (isa<top::WeightOp>(op.bias().getDefiningOp())) {
    bias_quant = cast<top::WeightOp>(op.bias().getDefiningOp()).read<int32_t>();
  } else {
    bias_quant = std::shared_ptr<std::vector<int32_t>>(
        new std::vector<int32_t>(col_size, 0));
  }
  auto bias_type = RankedTensorType::get({col_size}, rewriter.getI32Type());

  if (can_merge_izp) {
    //    attrs.push_back(rewriter.getNamedAttr(
    //        "multipliers", rewriter.getI64ArrayAttr(multiplier)));
    //    attrs.push_back(
    //        rewriter.getNamedAttr("rshifts",
    //        rewriter.getI64ArrayAttr(-shift)));
    //    attrs.push_back(rewriter.getNamedAttr(
    //        "quant_mode",
    //        tpu::RequantModeAttr::get(ctx, tpu::RequantMode::TFlite_Lshift)));
    if (input_zeroPoint) {
      // merge input_zeroPoint to bias
      std::shared_ptr<std::vector<int8_t>> right_quant;
      right_quant =
          cast<top::WeightOp>(op.right().getDefiningOp()).read<int8_t>();
      for (size_t r_ind = 0; r_ind < row_size; ++r_ind) {
        for (size_t c_ind = 0; c_ind < col_size; ++c_ind) {
          auto right_data = op.right_transpose()
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
      operands.push_back(op.bias());
    }
    auto matmul_type = RankedTensorType::get(Module::getShape(op.output()),
                                             rewriter.getI32Type());
    auto new_name = Module::getName(op.getOperation()).str() + "_matmul_no_izp";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfter(op);
    auto newOp =
        rewriter.create<tpu::MatMulOp>(name_loc, matmul_type, operands, attrs);
    // do requant
    auto newValue =
        do_requant(op->getLoc(), newOp.output(), op.output().getType(), true,
                   multiplier, shift, tpu::RequantMode::TFlite_Lshift);
    rewriter.replaceOp(op, {newValue});
    // rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, op.output().getType(),
    //                                            operands, attrs);
  } else {
    // (M * K) (K * N)
    // (Input - izp) Matmul (Right - kzp) ==> (Input) Matmul (Right - kzp) -
    // (izp) Matmul (Right - kzp)
    // - (izp) Matmul (Right - kzp) ==> izp * kzp * K - izp *
    // reduce_sum(Right.col) for each row

    // merge izp * kzp * K to bias
    for (size_t c_ind = 0; c_ind < col_size; ++c_ind) {
      bias_quant->data()[c_ind] +=
          input_zeroPoint * right_zero_point * row_size;
    }
    auto new_bias = top::WeightOp::create(op, "MergedInputZeroPoint",
                                          *bias_quant, bias_type);
    operands.push_back(new_bias);
    auto matmul_type = RankedTensorType::get(Module::getShape(op.output()),
                                             rewriter.getI32Type());
    auto new_name = Module::getName(op.getOperation()).str() + "_matmul_no_izp";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfter(op);
    auto newOp =
        rewriter.create<tpu::MatMulOp>(name_loc, matmul_type, operands, attrs);

    // rewriter.replaceOpWithNewOp<tpu::MatMulOp>(op, op.output().getType(),
    //                                            operands, attrs);
    // do reduce
    new_name = Module::getName(op.right()).str() + "_reduce_h";
    attrs.erase(attrs.begin(), attrs.end());
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({K_idx})));
    attrs.push_back(
        rewriter.getNamedAttr("keepdims", rewriter.getI64IntegerAttr(1)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("ReduceSum")));
    auto reduce_shape = std::vector<int64_t>(Module::getShape(op.right()));
    reduce_shape[K_idx] = 1;

    auto newType = RankedTensorType::get(reduce_shape, rewriter.getI32Type());
    if (op.right_transpose())
      newType = RankedTensorType::get(reduce_shape, rewriter.getI32Type());
    name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfterValue(op.right());
    auto reduceOp = rewriter.create<tpu::ReduceOp>(
        name_loc, newType,
        ValueRange{op.right(), Module::getNoneOp(op), Module::getNoneOp(op)},
        attrs);
    Value newValue = reduceOp.output();
    // do reshape
    attrs.erase(attrs.begin(), attrs.end());
    if (op.right_transpose()) {
      auto reshapeType =
          RankedTensorType::get({1, col_size}, rewriter.getI32Type());
      newValue = do_reshape(newValue, reshapeType);
    }
    // do mulconst
    newValue = do_binary_saclar<tpu::MulConstOp>(
        newValue, rewriter.getI32Type(), -input_zeroPoint);
    // do add
    new_name = Module::getName(newOp.output()).str() + "_add_zp";
    name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    rewriter.setInsertionPointAfterValue(newOp);
    auto addOp = rewriter.create<tpu::AddOp>(
        name_loc, matmul_type, ValueRange{newOp.output(), newValue}, attrs);
    // do requant
    newValue =
        do_requant(op->getLoc(), addOp.output(), op.output().getType(), true,
                   multiplier, shift, tpu::RequantMode::TFlite_Lshift);
    rewriter.replaceOp(op, {newValue});
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
