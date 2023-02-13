//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-add"

namespace tpu_mlir {
namespace cv18xx {
void AddLowering::LoweringINT8(PatternRewriter &rewriter, top::AddOp op,
                               bool asymmetric) const {
  std::vector<Value> operands;
  const int nInputs = op->getNumOperands();
  double o_scale;
  std::vector<int64_t> rshift_v(1);
  std::vector<int64_t> multiplier_v(nInputs, 1);
  std::vector<float> qscale(nInputs, 1.0);
  float max_qscale = 0.0;

  o_scale = module::getThreshold(op.getOutput());
  bool hasConst = false;
  for (int i = 1; i < nInputs; ++i) {
    if (isa<top::WeightOp>(op->getOperand(i).getDefiningOp())) {
      hasConst = true;
      break;
    }
  }
  bool isQuantBF16WhenConst = true;

  if (hasConst) {
    assert(nInputs == 2);
    if (isQuantBF16WhenConst) {
      return LoweringBF16(rewriter, op);
    } else {
      auto opd0 = op->getOperand(0);
      assert(!isa<top::WeightOp>(opd0.getDefiningOp()));
      double i_scale = module::getThreshold(opd0);

      int const_idx = 1;
      auto weightOp =
          cast<top::WeightOp>(op->getOperand(const_idx).getDefiningOp());
      auto const_f32 = weightOp.read<float>();
      auto const_shape = module::getShape(weightOp.getResult());
      auto const_size = const_f32->size();
      auto max_elem = findMaxabs(const_f32->data(), const_size);

      std::vector<int8_t> quant_const(const_size, 0);
      for (size_t i = 0; i < const_size; ++i) {
        float float_quant = const_f32->at(i) * 127.0 / max_elem;
        quant_const[i] = to_int8(float_quant, ROUNDING_HALF_UP);
      }

      auto weight_type = weightOp.getType().cast<RankedTensorType>();
      auto new_weight_type = RankedTensorType::get(
          weight_type.getShape(), rewriter.getIntegerType(8, true));
      auto weight_operand =
          top::WeightOp::create(op, "quant", quant_const, new_weight_type);
      operands.emplace_back(opd0);
      operands.emplace_back(weight_operand);

      // determine the qscale
      qscale[0] = i_scale / o_scale;
      qscale[1] = max_elem / o_scale;
      max_qscale = std::max(qscale[0], qscale[1]);
    }
  } else {
    auto coeff_v = module::getF64Array(op.getCoeff(), nInputs, 1.0);

    for (int i = 0; i < nInputs; i++) {
      auto input = op->getOperand(i);
      operands.push_back(input);
      double i_scale = module::getThreshold(input);
      auto scale_f = i_scale / o_scale;
      qscale[i] = coeff_v->at(i) * scale_f;
    }

    for (auto &q : qscale) {
      if (max_qscale < std::abs(q)) {
        max_qscale = std::abs(q);
      }
    }
  }
  int64_t multiplier = 0;
  int64_t shift = 0;
  getRShiftAndMultiplierFromQScale(max_qscale, &multiplier, &shift, false);

  rshift_v[0] = shift;
  for (int i = 0; i < nInputs; ++i) {
    multiplier_v[i] = getMultiplierI8FromQScaleAndRShift(qscale[i], shift);
  }

  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "multipliers", rewriter.getI64ArrayAttr(multiplier_v)));
  attrs.push_back(
      rewriter.getNamedAttr("rshifts", rewriter.getI64ArrayAttr(rshift_v)));
  auto newType = getQuantInt8Type(op.getOutput(), false);
  rewriter.replaceOpWithNewOp<tpu::AddOp>(op.getOperation(), newType, operands,
                                          attrs);
  return;
}
void AddLowering::LoweringBF16(PatternRewriter &rewriter, top::AddOp op) const {
  lowering_common_bf16<tpu::AddOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
