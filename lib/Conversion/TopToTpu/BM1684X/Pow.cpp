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

void PowTryLowering::Lowering(PatternRewriter &rewriter, top::PowOp op) const {

  // auto prev_op = op.getInput().getDefiningOp();
  // if (!prev_op->hasTrait<trait::ShapeProducer>()) {
  //   return;
  // }
  if (!isa_shape_subnet_op(op))
    return;

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("exponent", rewriter.getF32FloatAttr(op.getExponent().convertToDouble())));

  auto v = op.getOutput();
  Type new_type = RankedTensorType::get(module::getShape(v), rewriter.getF32Type());
  rewriter.replaceOpWithNewOp<tpu::ShapePowOp>(op, new_type, op->getOperands(), attrs);
}

/**
 * @note for the sake of avoiding x < 0 SINCE y = x ^ n is translated as e ^ (n
 * * log(x))
 *
 * y = → (if n even) {abs(x) ^ n;}
 *     ↘ (if n not even) → (if n is int) {x * abs(x) ^ (n - 1);}
 *                       ↘ (if n is not int) {e ^ (n * log(x));} → (if x >= 0)
 * {x ^ n;} ↘ (if x < 0) {nan}
 */
void PowLowering::LoweringF32(PatternRewriter &rewriter, top::PowOp op) const {
  auto replace_pow = [&rewriter](top::PowOp &op, double n) -> Value {
    auto name = module::getName(op.getOutput());
    auto type = op.getOutput().getType();
    rewriter.setInsertionPointAfter(op);
    auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
    std::vector<NamedAttribute> attrs;

    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
    auto log_op = rewriter.create<tpu::ActiveOp>(
        log_loc, type, ValueRange{op.getInput()}, attrs);
    auto mul_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_mul_const"));
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(n)));
    auto mul_op = rewriter.create<tpu::MulConstOp>(
        mul_loc, type, ValueRange{log_op.getOutput()}, attrs);
    auto ex_loc = op.getLoc();
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
    auto ex_op = rewriter.create<tpu::ActiveOp>(
        ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
    op.replaceAllUsesWith(ex_op.getOperation());
    return ex_op.getOutput();
  };

  auto insert_abs = [&rewriter](top::PowOp &op) -> tpu::ActiveOp {
    auto name = module::getName(op.getOutput());
    std::vector<NamedAttribute> attrs;
    auto abs_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_abs"));
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::ABSVAL)));
    auto abs_op = rewriter.create<tpu::ActiveOp>(
        abs_loc, op.getOutput().getType(), ValueRange{op.getInput()}, attrs);
    op->setOperand(0, abs_op.getOutput());
    return abs_op;
  };

  double exponent = op.getExponent().convertToDouble();

  if (fmod(exponent, 2) == 0) {
    insert_abs(op);
    replace_pow(op, exponent);

    rewriter.eraseOp(op);
  } else {
    if ((int)exponent == exponent) {
      auto abs_op = insert_abs(op);
      Value v_replaced = replace_pow(op, exponent - 1);

      // insert mul
      std::vector<NamedAttribute> attrs;
      auto name = module::getName(op.getOutput());
      auto exp_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_exp"));
      v_replaced.getDefiningOp()->setLoc(exp_loc);
      auto x = abs_op.getInput();
      auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str()));
      std::vector<Value> mul_operands;
      mul_operands.push_back(x);
      mul_operands.push_back(v_replaced);
      auto mul_op = rewriter.create<tpu::MulOp>(
          mul_loc, op.getOutput().getType(), mul_operands, attrs);
      v_replaced.replaceAllUsesExcept(mul_op.getOutput(), mul_op);
      rewriter.eraseOp(op);
    } else {
      replace_pow(op, exponent);
      rewriter.eraseOp(op);
    }
    return;
  }
}

static double g_ex = 0;
void PowLowering::LoweringINT8(PatternRewriter &rewriter, top::PowOp op,
                               bool asymmetric) const {
  g_ex = op.getExponent().convertToDouble();
  auto table =
      create_lookup_table(op.getInput(), op.getOutput(), asymmetric,
                          [](double val) { return std::pow(val, g_ex); });
  auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
  rewriter.replaceOpWithNewOp<tpu::LutOp>(op, newType,
                                          ValueRange{op.getInput(), table});
}

void PowLowering::LoweringINT4(PatternRewriter &rewriter, top::PowOp op,
                                   bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void PowLowering::LoweringBF16(PatternRewriter &rewriter, top::PowOp op) const {
  // support mars3 bf16
  auto replace_pow = [&rewriter](top::PowOp &op, double n) -> Value {
    auto name = module::getName(op.getOutput());
    // auto type = op.getOutput().getType();
    // if(module::isMARS3())
    auto type = module::isMARS3() ? getQuantBF16Type(op->getResult(0)) : op.getOutput().getType();
    rewriter.setInsertionPointAfter(op);
    auto log_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_log"));
    std::vector<NamedAttribute> attrs;
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::LN)));
    auto log_op = rewriter.create<tpu::ActiveOp>(
        log_loc, type, ValueRange{op.getInput()}, attrs);

    auto mul_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_mul_const"));
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(n)));
    auto mul_op = rewriter.create<tpu::MulConstOp>(
        mul_loc, type, ValueRange{log_op.getOutput()}, attrs);
    auto ex_loc = op.getLoc();
    attrs.clear();
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::EXP)));
    auto ex_op = rewriter.create<tpu::ActiveOp>(
        ex_loc, type, ValueRange{mul_op.getOutput()}, attrs);
    op.replaceAllUsesWith(ex_op.getOperation());
    return ex_op.getOutput();
  };
  auto insert_abs = [&rewriter](top::PowOp &op) -> tpu::ActiveOp {
    auto name = module::getName(op.getOutput());
    std::vector<NamedAttribute> attrs;
    auto abs_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_abs"));
    attrs.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::ActiveModeAttr::get(op.getContext(), tpu::ActiveMode::ABSVAL)));
    auto new_type = getQuantBF16Type(op.getResult());
    auto abs_op = rewriter.create<tpu::ActiveOp>(
        abs_loc, new_type, ValueRange{op.getInput()}, attrs);
    op->setOperand(0, abs_op.getOutput());
    return abs_op;
  };
  // support f32, change type when needed
  double exponent = op.getExponent().convertToDouble();

  if (fmod(exponent, 2) == 0) {
    insert_abs(op);
    replace_pow(op, exponent);

    rewriter.eraseOp(op);
  } else {
    if ((int)exponent == exponent) {
      auto abs_op = insert_abs(op);
      Value v_replaced = replace_pow(op, exponent - 1);

      // insert mul
      std::vector<NamedAttribute> attrs;
      auto name = module::getName(op.getOutput());
      auto exp_loc = NameLoc::get(rewriter.getStringAttr(name.str() + "_exp"));
      v_replaced.getDefiningOp()->setLoc(exp_loc);
      auto x = abs_op.getInput();
      auto mul_loc = NameLoc::get(rewriter.getStringAttr(name.str()));
      std::vector<Value> mul_operands;
      mul_operands.push_back(x);
      mul_operands.push_back(v_replaced);
      auto mul_op = rewriter.create<tpu::MulOp>(
          mul_loc, op.getOutput().getType(), mul_operands, attrs);
      v_replaced.replaceAllUsesExcept(mul_op.getOutput(), mul_op);
      rewriter.eraseOp(op);
    } else {
      replace_pow(op, exponent);
      rewriter.eraseOp(op);
    }
    return;
  }
}

void PowLowering::LoweringF16(PatternRewriter &rewriter, top::PowOp op) const {
  LoweringF32(rewriter, op);
}

void PowLowering::LoweringF8(PatternRewriter &rewriter, top::PowOp op) const {
  UNREACHABLE_OP("Not Implemented", op);
}

void PowLowering::LoweringQuantized(PatternRewriter &rewriter,
                                    top::PowOp op) const {
  // UNREACHABLE_OP("Not Implemented", op);
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
