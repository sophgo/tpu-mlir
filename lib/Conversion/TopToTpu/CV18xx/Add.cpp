//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

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
  for (int i = 0; i < nInputs; ++i) {
    if (module::isWeight(op->getOperand(i))) {
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
      assert(!module::isWeight(opd0));
      double i_scale = module::getThreshold(opd0);

      int const_idx = 1;
      auto weightOp =
          cast<top::WeightOp>(op->getOperand(const_idx).getDefiningOp());
      auto const_f32 = weightOp.read<float>();
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
  const int nInputs = op->getNumOperands();
  bool hasConst = false;
  int const_idx = -1;
  for (int i = 0; i < nInputs; ++i) {
    if (module::isWeight(op->getOperand(i))) {
      hasConst = true;
      const_idx = i;
      break;
    }
  }
  if (hasConst) {
    // Begin: here is special for sherpa
    int active_idx = 1 - const_idx;
    auto weightOp = op.getOperand(const_idx);
    auto activeOp = op.getOperand(active_idx);
    auto weight_shape = module::getShape(weightOp);
    auto active_shape = module::getShape(activeOp);
    int ndims = weight_shape.size();
    int active_ndims = active_shape.size();
    bool bcast_weight = false;
    bool bcast_active = false;
    if (ndims == active_ndims) {
      for (int i = 0; i < ndims; i++) {
        if (weight_shape[i] != active_shape[i]) {
          if (weight_shape[i] == 1) {
            bcast_weight = true;
          }
          if (active_shape[i] == 1) {
            bcast_active = true;
          }
        }
      }
    }
    bool need_bcast = bcast_weight && bcast_active;
    if (need_bcast && weightOp.hasOneUse()) {
      // broadcast operand0
      int bcast_dim = -1, dst_len = -1;
      for (int i = 0; i < ndims; i++) {
        if (weight_shape[i] != active_shape[i]) {
          if (weight_shape[i] == 1) {
            bcast_dim = i;
            dst_len = active_shape[i];
            break;
          }
        }
      }
      if (bcast_dim != -1) {
        llvm::errs() << "========BroadCast AddOp's Weight Operand==========\n";
        // llvm::errs()<<"active_dims="<<active_ndims<<",weight_ndims="<<ndims<<"\n";
        // op.dump();
        // broadcast weight
        auto weightOp =
            cast<top::WeightOp>(op->getOperand(const_idx).getDefiningOp());
        auto const_f32 = weightOp.read<float>();
        auto const_size = const_f32->size();
        std::vector<float> bcast_weight_const(const_size * dst_len);
        int once_bcast_len = 1;
        for (int i = bcast_dim + 1; i < ndims; i++) {
          once_bcast_len *= weight_shape[i];
        }
        int outer_dim = 1;
        for (int i = 0; i < bcast_dim; i++) {
          outer_dim *= weight_shape[i];
        }
        for (int i = 0; i < outer_dim; i++) {
          int src_idx = i * once_bcast_len;
          for (int j = 0; j < dst_len; j++) {
            int dst_idx = i * dst_len * once_bcast_len + j * once_bcast_len;
            memcpy(bcast_weight_const.data() + dst_idx,
                   const_f32->data() + src_idx, once_bcast_len * sizeof(float));
          }
        }
        auto new_weight_op_name =
            module::getName(weightOp.getResult()).str() + "_bcast";
        auto elt_type =
            weightOp.getType().cast<RankedTensorType>().getElementType();
        std::vector<int64_t> new_shape(ndims);
        for (int i = 0; i < ndims; i++) {
          new_shape[i] = (i == bcast_dim) ? dst_len : weight_shape[i];
        }
        auto new_weight_type = RankedTensorType::get(new_shape, elt_type);
        auto new_weight_operand = top::WeightOp::create(
            op, new_weight_op_name, bcast_weight_const, new_weight_type);
        op->setOperand(const_idx, new_weight_operand);
        // op->setOperand(1, activeOp);
      }
    }
  }
  lowering_common_bf16<tpu::AddOp>(rewriter, op);
}

} // namespace cv18xx
} // namespace tpu_mlir
