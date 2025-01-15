//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"

using namespace tpu_mlir::backend;

namespace tpu_mlir {
namespace bm1684x {

static void tans_shape(std::vector<int64_t> &order, int64_t axis, bool front) {
  for (int i = 0; i < 4; i++) {
    if (i == 1) {
      order.push_back(front ? axis : 1);
    } else if (i == axis) {
      order.push_back(front ? 1 : axis);
    } else {
      order.push_back(i);
    }
  }
  return;
}

static mlir::RankedTensorType
getQuantInt8TypeNewShape(Value v, std::vector<int64_t> new_shape,
                         bool asymmetric) {
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
  return RankedTensorType::get(new_shape, qtype);
}

void SoftmaxLowering::LoweringF32(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  if (module::isMARS3() || module::isSGTPUV8())
    lowering_common_bf16<tpu::SoftmaxOp>(rewriter, op, 6);
  else
    lowering_common_f32<tpu::SoftmaxOp>(rewriter, op, 6);
}
void SoftmaxLowering::LoweringINT4(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}
void SoftmaxLowering::LoweringINT8(PatternRewriter &rewriter, top::SoftmaxOp op,
                                   bool asymmetric) const {
  auto in_shape = module::getShape(op);
  std::vector<int64_t> new_shape(in_shape);
  bool need_reshape = false;
  auto dims = in_shape.size();

  if (op.getLog() || dims > 4 || op.getAxis() != 1) {
    if (!module::isMARS3() && !module::isSGTPUV8())
      return LoweringF16(rewriter, op);
    else
      return LoweringBF16(rewriter, op);
  }

  if (dims < 4) {
    for (int i = dims; i < 4; i++) {
      new_shape.push_back(1);
    }
    need_reshape = true;
  }

  auto in_reshaped_type =
      getQuantInt8TypeNewShape(op.getInput(), new_shape, asymmetric);
  auto out_type = getQuantInt8Type(op.getOutput(), asymmetric);
  auto out_ttype =
      RankedTensorType::get(in_shape, module::getStorageType(out_type));

  auto in = op.getOperand();
  double scale;
  int64_t zeropoint;
  auto beta_v = op.getBeta().convertToDouble();
  module::getScaleAndZeroPoint(in, scale, zeropoint, asymmetric);
  std::vector<float> table(256, 0.0f);
  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(-1.0 * scale * i * beta_v);
  }
  auto table_opd = create_lookup_table(op, table);

  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "axis" && op.getAxis() != 1) {
      attrs.push_back(
          rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(1)));
    } else
      attrs.push_back(attr);
  }

  if (op.getAxis() == 1) {
    if (need_reshape) {
      auto ctx = op.getInput().getContext();
      OpBuilder builder(ctx);
      auto in_reshaped = do_reshape(op.getInput(), in_reshaped_type);

      builder.setInsertionPointAfterValue(in_reshaped);
      auto new_name = (module::getName(op.getOperation())).str() + "__softmax";
      auto sftmax_name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      auto sft_out_type =
          getQuantInt8TypeNewShape(op.getOutput(), new_shape, asymmetric);
      auto newOp = rewriter.create<tpu::SoftmaxOp>(
          sftmax_name_loc, sft_out_type,
          ValueRange{in_reshaped, table_opd,
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation())},
          attrs);

      auto reshaped_out = do_reshape(newOp.getOutput(), out_ttype);
      builder.setInsertionPointAfterValue(reshaped_out);
      rewriter.replaceOp(op, {reshaped_out});
    } else {
      rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
          op, out_type,
          ValueRange{op.getInput(), table_opd,
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation())},
          attrs);
    }
  } else {
    // dead code
    auto ctx = op.getInput().getContext();
    OpBuilder builder(ctx);

    auto handle_transposed = [&](mlir::Value in) {
      std::vector<int64_t> order_in;
      tans_shape(order_in, op.getAxis(), true);
      std::string new_name =
          module::getName(op.getInput()).str() + "__transpose";
      auto in_trans_name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      auto trans_in_op = do_transpose(in_trans_name_loc, in, order_in);

      // builder.setInsertionPointAfterValue(trans_in_op);
      new_name = (module::getName(op.getOperation())).str() + "__softmax";
      auto sftmax_name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      auto sft_out_type = getQuantInt8TypeNewShape(
          op.getOutput(), module::getShape(trans_in_op), asymmetric);
      auto newOp = rewriter.create<tpu::SoftmaxOp>(
          sftmax_name_loc, sft_out_type,
          ValueRange{trans_in_op, table_opd,
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation()),
                     module::getNoneOp(op.getOperation())},
          attrs);

      new_name = (module::getName(op.getOutput()).str()) + "__transpose";
      auto out_trans_name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      std::vector<int64_t> order_out;
      tans_shape(order_out, op.getAxis(), true);
      auto trans_out_op =
          do_transpose(out_trans_name_loc, newOp.getOutput(), order_out);
      return trans_out_op;
    };

    if (need_reshape) {
      auto in_reshaped = do_reshape(op.getInput(), in_reshaped_type);
      auto trans_out_op = handle_transposed(in_reshaped);
      auto reshaped_out = do_reshape(trans_out_op, out_ttype);
      builder.setInsertionPointAfterValue(reshaped_out);
      rewriter.replaceOp(op, {reshaped_out});
    } else {
      auto trans_out_op = handle_transposed(op.getInput());
      builder.setInsertionPointAfterValue(trans_out_op);
      rewriter.replaceOp(op, {trans_out_op});
    }
  }
}

static bool axis_dim_too_large(top::SoftmaxOp op) {
  auto max_dim = Arch::LMEM_BANK_BYTES / 2; // bf16 or f16
  auto in_shape = module::getShape(op.getInput());
  auto axis_dim = in_shape[op.getAxis()];
  return axis_dim > max_dim;
}

void SoftmaxLowering::LoweringBF16(PatternRewriter &rewriter,
                                   top::SoftmaxOp op) const {
  if (axis_dim_too_large(op)) {
    // ugly code, if nodechip_softmax_local_1x1 support f16 and bf16 in future,
    // remove code here
    LoweringF32(rewriter, op);
    return;
  }
  lowering_common_bf16<tpu::SoftmaxOp>(rewriter, op, 6);
}

void SoftmaxLowering::LoweringF16(PatternRewriter &rewriter,
                                  top::SoftmaxOp op) const {
  if (axis_dim_too_large(op)) {
    // ugly code, if nodechip_softmax_local_1x1 support f16 and bf16 in future,
    // remove code here
    LoweringF32(rewriter, op);
    return;
  }
  lowering_common_f16<tpu::SoftmaxOp>(rewriter, op, 6);
}

void SoftmaxLowering::LoweringF8(PatternRewriter &rewriter,
                                 top::SoftmaxOp op) const {
  LoweringF32(rewriter, op);
}

void SoftmaxLowering::LoweringQuantized(PatternRewriter &rewriter,
                                        top::SoftmaxOp op) const {
  if (module::isUniformQuantized(op.getInput(), op.getOutput()) == false) {
    llvm_unreachable("input output should be quantized");
  }
  int64_t zeropoint;
  double i_scale;
  module::getScaleAndZeroPoint(op.getInput(), i_scale, zeropoint, true);
  std::vector<float> table(256, 0.0f);
  auto beta_v = op.getBeta().convertToDouble();
  auto scale = -i_scale * beta_v;

  // const int int_bits = 5;
  // const double_t multiplier_max = (1ULL << 31) - 1;
  // double_t multiplier_real =
  //     std::min(i_scale * beta_v * (1 << (31 - int_bits)), multiplier_max);
  // int32_t multi, shift;
  // QuantizeMultiplier(multiplier_real, &multi, &shift);
  // double max_input_rescaled =
  //     1.0 * ((1 << int_bits) - 1) * (1LL << (31 - int_bits)) / (1LL <<
  //     shift);
  // int32_t diff_min = -1 *
  // static_cast<int32_t>(std::floor(max_input_rescaled)); std::vector<int32_t>
  // table(256, 0); for (int i = 0; i < 256; ++i) {
  //   int32_t input_diff_rescaled = MultiplyByQuantizedMultiplier(-i, multi,
  //   shift); table[i] = exp_on_negative_values(input_diff_rescaled, int_bits);
  // }

  for (int i = 0; i < 256; ++i) {
    table[i] = std::exp(scale * i);
  }
  auto table_opd = create_lookup_table(op, table);

  auto axis = op.getAxis();
  std::vector<NamedAttribute> attrs;
  for (auto &attr : op->getAttrs()) {
    if (attr.getName() == "axis" && axis != 1) {
      attrs.push_back(
          rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(1)));
    } else if (attr.getName() == "round_mode") {
      auto round_mode = get_round_mode(op.getRoundModeAttr().str());
      attrs.push_back(rewriter.getNamedAttr(
          "round_mode", tpu::RoundModeAttr::get(op.getContext(), round_mode)));
    } else
      attrs.push_back(attr);
  }
  if (axis == 1) {
    rewriter.replaceOpWithNewOp<tpu::SoftmaxOp>(
        op, op.getOutput().getType(),
        ValueRange{op.getInput(), table_opd,
                   module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation())},
        attrs);
  } else if (axis > 1) {
    // transpose
    auto dims = module::getShape(op.getInput()).size();
    std::string new_name = module::getName(op.getInput()).str() + "__transpose";
    auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    std::vector<int64_t> order(1, 0);
    order.push_back(axis);
    for (int i = 1; i < axis; i++) {
      order.push_back(i);
    }
    for (int i = axis + 1; i < dims; i++) {
      order.push_back(i);
    }
    auto TransOp = do_transpose(name_loc, op.getInput(), order);
    // softmax
    rewriter.setInsertionPointAfter(op);
    new_name = (module::getName(op.getOutput()).str()) + "__softmax";
    name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
    auto newType =
        module::getTypeLike(op.getOutput(), module::getShape(TransOp));
    auto newOp = rewriter.create<tpu::SoftmaxOp>(
        name_loc, newType,
        ValueRange{TransOp, table_opd, module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation()),
                   module::getNoneOp(op.getOperation())},
        attrs);
    // transpose
    std::vector<int64_t> order1(1, 0);
    for (int i = 1; i < axis; i++) {
      order1.push_back(i + 1);
    }
    order1.push_back(1);
    for (int i = axis + 1; i < dims; i++) {
      order1.push_back(i);
    }
    auto v = do_transpose(op->getLoc(), newOp, order1);
    rewriter.replaceOp(op, {v});
  } else {
    UNREACHABLE_OP("Not Implemented", op);
  }
}

} // namespace bm1684x
} // namespace tpu_mlir
