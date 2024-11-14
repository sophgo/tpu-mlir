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

#define FP16_MAX 65504.0
#define FP16_MIN -65504.0
#define BF16_MAX 3.3895314e38
#define BF16_MIN -3.3895314e38

namespace tpu_mlir {
namespace bm1684x {

void DivConstTryLowering::Lowering(PatternRewriter &rewriter,
                                   top::DivConstOp op) const {
  auto prev_op = op.getInput().getDefiningOp();
  if (!prev_op->hasTrait<trait::ShapeProducer>()) {
    return;
  }
  if (op.getIsReverse()) {
    llvm_unreachable("Not implemented");
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("type", rewriter.getStringAttr("Div")));

  auto constI32 = i32_array_t(new std::vector<int32_t>(1, 0));
  constI32->data()[0] = std::floor(op.getConstVal().convertToDouble());
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

// void DivConstLowering::LoweringF32(PatternRewriter &rewriter,
//                                    top::DivConstOp op) const {
//   lowering_common_f32<tpu::DivConstOp>(rewriter, op);
// }
// void DivConstLowering::LoweringINT4(PatternRewriter &rewriter,
// top::DivConstOp op,
//                                    bool asymmetric) const {
//   LoweringINT8(rewriter, op, asymmetric);
// }
// void DivConstLowering::LoweringINT8(PatternRewriter &rewriter,
//                                     top::DivConstOp op, bool asymmetric)
//                                     const {
//   double scale_i, scale_o;
//   int64_t zp_i, zp_o;
//   module::getScaleAndZeroPoint(op.getInput(), scale_i, zp_i, asymmetric);
//   module::getScaleAndZeroPoint(op.getOutput(), scale_o, zp_o, asymmetric);
//   auto scale = scale_i / scale_o * op.getConstVal().convertToDouble();
//   int multiplier, rshift;
//   get_scale_and_shift(scale, multiplier, rshift, 8);
//   std::vector<NamedAttribute> attrs;
//   for (auto &attr : op->getAttrs()) {
//     attrs.push_back(attr);
//   }

//   attrs.push_back(rewriter.getNamedAttr(
//       "multiplier", rewriter.getSI32IntegerAttr(multiplier)));
//   attrs.push_back(
//       rewriter.getNamedAttr("rshift", rewriter.getSI32IntegerAttr(rshift)));
//   auto newType = getQuantInt8Type(op.getOutput(), asymmetric);
//   rewriter.replaceOpWithNewOp<tpu::MulShiftOp>(op, newType,
//                                                ValueRange{op.getInput()},
//                                                attrs);
// }

// void DivConstLowering::LoweringBF16(PatternRewriter &rewriter,
//                                     top::DivConstOp op) const {
//   auto const_v = op.getConstVal().convertToDouble();
//   if (const_v > BF16_MAX || const_v < BF16_MIN)
//     LoweringF32(rewriter, op);
//   else
//     lowering_common_bf16<tpu::DivConstOp>(rewriter, op);
// }

// void DivConstLowering::LoweringF16(PatternRewriter &rewriter,
//                                    top::DivConstOp op) const {
//   auto const_v = op.getConstVal().convertToDouble();
//   if (const_v > FP16_MAX || const_v < FP16_MIN)
//     LoweringF32(rewriter, op);
//   else
//     lowering_common_f16<tpu::DivConstOp>(rewriter, op);
// }

// void DivConstLowering::LoweringF8(PatternRewriter &rewriter,
//                                    top::DivConstOp op) const {
//   std::vector<NamedAttribute> attrs;
//   double const_v = op.getConstVal().convertToDouble();
//   auto qtype_in = module::getCalibratedType(op.getInput());
//   auto qtype_out = module::getCalibratedType(op.getOutput());

//   bool isE4 = module::getMode() == module::Mode::F8E4M3;
//   if (isE4) {
//     double in_scale = qtype_in.getMax() / get_f8e4m3_max();
//     double out_scale = qtype_out.getMax() / get_f8e4m3_max();
//     const_v = const_v * in_scale / out_scale;
//   }
//   for (auto &attr : op->getAttrs()) {
//     if (attr.getName() == "const_val") {
//       attrs.push_back(rewriter.getNamedAttr("const_val",
//       rewriter.getF64FloatAttr(const_v)));
//     } else {
//       attrs.push_back(attr);
//     }
//   }

//   if (isE4) {
//     auto newType = getQuantF8E4M3Type(op.getOutput());
//     rewriter.replaceOpWithNewOp<tpu::DivConstOp>(op, newType,
//                                                ValueRange{op.getInput()},
//                                                attrs);
//   } else {
//     auto newType = getQuantF8E5M2Type(op.getOutput());
//     rewriter.replaceOpWithNewOp<tpu::DivConstOp>(op, newType,
//                                                ValueRange{op.getInput()},
//                                                attrs);
//   }
// }

// void DivConstLowering::LoweringQuantized(PatternRewriter &rewriter,
//                                          top::DivConstOp op) const {
//   // UNREACHABLE_OP("Not Implemented", op);
//   LoweringINT8(rewriter, op, true);
// }

} // namespace bm1684x
} // namespace tpu_mlir
