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

static std::string mode_convert(llvm::StringRef mode) {
  if (mode.compare("Equal") == 0) {
    return "Equal";
  } else if (mode.compare("Greater") == 0) {
    return "Greater";
  } else if (mode.compare("GreaterOrEqual") == 0) {
    return "GreaterOrEqual";
  } else if (mode.compare("Less") == 0) {
    return "Less";
  } else if (mode.compare("LessOrEqual") == 0) {
    return "LessOrEqual";
  } else if (mode.compare("NotEqual") == 0) {
    return "NotEqual";
  } else if (mode.compare("And") == 0) {
    return "And";
  } else if (mode.compare("Not") == 0) {
    return "Not";
  }
  return "";
}

void CompareConstTryLowering::Lowering(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  auto prev_op = op.getInput().getDefiningOp();
  if (!prev_op->hasTrait<trait::ShapeProducer>()) {
    return;
  }
  if (op.getInversed()) {
    llvm_unreachable("not implement");
  }
  auto compare_mode = op.getMode();
  auto converted_mode = mode_convert(compare_mode);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("type", rewriter.getStringAttr(converted_mode)));
  auto constI32 = i32_array_t(new std::vector<int32_t>(1, 0));
  constI32->data()[0] =
      static_cast<int64_t>(op.getConstVal().convertToDouble());
  auto weight_type =
      RankedTensorType::get({1}, rewriter.getIntegerType(32, true));
  auto weight_op = top::WeightOp::create(op, "i64", *constI32, weight_type);
  std::vector<Value> operands;
  operands.push_back(op.getInput());
  operands.push_back(weight_op);
  Type new_type =
      RankedTensorType::get(module::getShape(op.getOutput()),
                            IntegerType::get(op.getOutput().getContext(), 32));
  rewriter.replaceOpWithNewOp<tpu::ShapeArithOp>(op, new_type, operands, attrs);
}

void CompareConstLowering::LoweringF32(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  lowering_common_f32<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringINT4(PatternRewriter &rewriter,
                                        top::CompareConstOp op,
                                        bool asymmetric) const {
  LoweringINT8(rewriter, op, asymmetric);
}

void CompareConstLowering::LoweringINT8(PatternRewriter &rewriter,
                                        top::CompareConstOp op,
                                        bool asymmetric) const {
  if (op.getMode().str() == "And") {
    if (module::isMARS3() || module::isSGTPUV8()) {
      lowering_common_bf16<tpu::CompareConstOp>(rewriter, op);
    } else {
      lowering_common_f16<tpu::CompareConstOp>(rewriter, op);
    }
  } else {
    auto op_ = op.getOperation();
    int64_t zp;
    double scale;
    bool sign;
    module::getScaleAndZeroPoint(op.getInput(), scale, zp, sign, asymmetric);
    auto val = op.getConstVal().convertToDouble();
    double new_val = std::round(val / scale + zp);
    new_val = sign ? to_int8(new_val) : to_uint8(new_val);
    op_->setAttr("const_val", rewriter.getF64FloatAttr(new_val));
    auto newType = getQuantBoolType(op.getOutput());
    lowering_common<tpu::CompareConstOp>(rewriter, op.getOperation(), newType);
  }
}

void CompareConstLowering::LoweringBF16(PatternRewriter &rewriter,
                                        top::CompareConstOp op) const {
  lowering_common_bf16<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringF16(PatternRewriter &rewriter,
                                       top::CompareConstOp op) const {
  lowering_common_f16<tpu::CompareConstOp>(rewriter, op);
}

void CompareConstLowering::LoweringF8(PatternRewriter &rewriter,
                                      top::CompareConstOp op) const {
  llvm_unreachable("FIXME: not implement");
}

void CompareConstLowering::LoweringQuantized(PatternRewriter &rewriter,
                                             top::CompareConstOp op) const {
  LoweringINT8(rewriter, op, true);
}

} // namespace bm1684x
} // namespace tpu_mlir
