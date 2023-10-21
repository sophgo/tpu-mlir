//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/Conversion.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include <regex>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

template <typename OpTy> static void BackwardOp(OpTy op) {
  Value in = op.getInput();
  Value out = op.getOutput();
  auto new_type = module::getTypeLike(out, module::getShape(in));
  in.setType(new_type);
}

static void Backward(Value in) {
  if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
    BackwardOp(reshapeOp);
    // Backward(reshapeOp.getInput());
  } else if (auto permuteOp = dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
    BackwardOp(permuteOp);
    // Backward(permuteOp.getInput());
  } else if (auto d2s = dyn_cast<top::Depth2SpaceOp>(in.getDefiningOp())) {
    BackwardOp(d2s);
  }
}

template <typename OpTy> static void ForwardOp(OpTy op) {
  Value in = op.getInput();
  Value out = op.getOutput();
  auto new_type = module::getTypeLike(in, module::getShape(out));
  out.setType(new_type);
}

static void Forward(Value out) {
  for (auto user : out.getUsers()) {
    if (auto reshapeOp = dyn_cast<top::ReshapeOp>(user)) {
      ForwardOp(reshapeOp);
      // Forward(reshapeOp.getOutput());
    } else if (auto permuteOp = dyn_cast<top::PermuteOp>(user)) {
      ForwardOp(permuteOp);
      // Forward(permuteOp.getOutput());
    }
  }
}

template <typename TyOp>
struct ForwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<TyOp, top::ReduceOp>) {
      std::string mode = op.getMode().str();
      if (mode != "ReduceMax" && mode != "ReduceMin") {
        return failure();
      }
    }
    Value in = op.getInput();
    Value out = op.getOutput();
    if (!module::isCalibratedType(in)) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    if (module::isCalibratedType(out)) {
      auto out_qtype = module::getCalibratedType(out);
      if (in_qtype.getMax() == out_qtype.getMax() &&
          in_qtype.getMin() == out_qtype.getMin()) {
        return failure();
      }
    }
    auto new_type = RankedTensorType::get(module::getShape(out), in_qtype);
    out.setType(new_type);
    Forward(out);
    return success();
  }
};

struct ForwardMulConst : public OpRewritePattern<top::MulConstOp> {
  using OpRewritePattern<top::MulConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::MulConstOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getInput();
    Value out = op.getOutput();
    if (!module::isCalibratedType(in)) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    auto const_v = op.getConstVal().convertToDouble();
    auto in_min = in_qtype.getMin();
    auto in_max = in_qtype.getMax();
    auto out_max = (const_v >= 0 ? in_max : in_min);
    auto out_min = (const_v >= 0 ? in_min : in_max);
    if (const_v != (double)0) {
      out_max *= const_v;
      out_min *= const_v;
    }
    if (module::isCalibratedType(out)) {
      auto out_qtype = module::getCalibratedType(out);
      if (out_max == out_qtype.getMax() && out_min == out_qtype.getMin()) {
        return failure();
      }
    }
    auto new_out_type = quant::CalibratedQuantizedType::get(
        module::getStorageType(out), out_min, out_max);
    auto new_type = RankedTensorType::get(module::getShape(out), new_out_type);
    out.setType(new_type);
    Forward(out);
    return success();
  }
};

struct ForwardArg : public OpRewritePattern<top::ArgOp> {
  using OpRewritePattern<top::ArgOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::ArgOp op,
                                PatternRewriter &rewriter) const override {
    if (module::isNone(op.getValues())) {
      return failure();
    }
    auto in = op.getInput();
    auto out = op.getValues();
    if (!module::isCalibratedType(in)) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    if (module::isCalibratedType(out)) {
      auto out_qtype = module::getCalibratedType(out);
      if (in_qtype.getMax() == out_qtype.getMax() &&
          in_qtype.getMin() == out_qtype.getMin()) {
        return failure();
      }
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
    Forward(out);
    return success();
  }
};

template <typename TyOp>
struct KeepSignPattern : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getInput();
    Value out = op.getOutput();
    if (!module::isCalibratedType(in, out)) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    auto out_qtype = module::getCalibratedType(out);
    float min;
    if (in_qtype.getMin() < 0) {
      if (out_qtype.getMin() < 0) {
        return failure();
      }
      min = -out_qtype.getMax() * 0.1;
    } else {
      if (out_qtype.getMin() >= 0) {
        return failure();
      }
      min = 0;
    }
    auto etype = module::getStorageType(out);
    auto new_qtype =
        quant::CalibratedQuantizedType::get(etype, min, out_qtype.getMax());
    auto new_type = RankedTensorType::get(module::getShape(out), new_qtype);
    out.setType(new_type);
    Forward(out);
    return success();
  }
};

struct KeepAddSignPattern : public OpRewritePattern<top::AddOp> {
  using OpRewritePattern<top::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::AddOp op,
                                PatternRewriter &rewriter) const override {
    bool is_sign = false;
    auto num_inputs = op.getInputs().size();
    auto coeffs = module::getF64Array(op.getCoeff(), num_inputs, 1.0);
    for (int i = 0; i < num_inputs; i++) {
      auto in = op.getInputs()[i];
      auto coeff = coeffs->at(i);
      if (!module::isCalibratedType(in)) {
        return failure();
      }
      auto in_qtype = module::getCalibratedType(in);
      if (in_qtype.getMin() * coeff < 0 || in_qtype.getMax() * coeff < 0) {
        is_sign = true;
        break;
      }
    }
    auto out = op.getOutput();
    auto out_qtype = module::getCalibratedType(out);
    double min = out_qtype.getMin();
    if (is_sign && min >= 0) {
      min = -out_qtype.getMax() * 0.1;
    } else if (is_sign == false && min < 0) {
      min = 0;
    } else {
      return failure();
    }
    auto etype = module::getStorageType(out);
    auto new_qtype =
        quant::CalibratedQuantizedType::get(etype, min, out_qtype.getMax());
    auto new_type = RankedTensorType::get(module::getShape(out), new_qtype);
    out.setType(new_type);
    Forward(out);
    return success();
  }
};

struct SetSubConstSignPattern : public OpRewritePattern<top::SubConstOp> {
  using OpRewritePattern<top::SubConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::SubConstOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getInput();
    Value out = op.getOutput();
    if (!module::isCalibratedType(in) || !module::isCalibratedType(out)) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    if (module::isCalibratedType(out)) {
      auto out_qtype = module::getCalibratedType(out);
      auto out_type = out.getType().cast<RankedTensorType>();
      if (in_qtype.getMin() >= 0 && out_qtype.getMin() >= 0) {
        auto new_out_type = quant::CalibratedQuantizedType::get(
            module::getStorageType(out), out_qtype.getMax() * (-0.1),
            out_qtype.getMax());
        auto new_type =
            RankedTensorType::get(out_type.getShape(), new_out_type);
        out.setType(new_type);
        Forward(out);
        return success();
      } else {
        return failure();
      }
    }
    return failure();
  }
};

template <typename TyOp, bool KeepMin = false>
struct BackwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op->getOperand(0);
    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }
    if (in.hasOneUse() == false) {
      return failure();
    }
    auto in_qtype = module::getCalibratedType(in);
    auto out_qtype = module::getCalibratedType(out);
    if (module::isCalibratedType(in)) {
      auto in_qtype = module::getCalibratedType(in);
      if (in_qtype.getMax() == out_qtype.getMax() &&
          (KeepMin || in_qtype.getMin() == out_qtype.getMin())) {
        return failure();
      }
    }
    auto in_type = in.getType().cast<RankedTensorType>();
    if (KeepMin) {
      auto etype = module::getStorageType(out);
      out_qtype = quant::CalibratedQuantizedType::get(etype, in_qtype.getMin(),
                                                      out_qtype.getMax());
    }
    auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
    in.setType(new_type);
    Backward(in);
    return success();
  }
};

template <typename TyOp>
struct ForwardTypePattern : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.getInput();
    Value out = op.getOutput();
    auto in_type = in.getType().cast<RankedTensorType>();
    auto out_type = out.getType().cast<RankedTensorType>();
    auto in_etype = in_type.getElementType();
    auto out_etype = out_type.getElementType();
    if (in_etype == out_etype) {
      return failure();
    }
    auto new_type = RankedTensorType::get(out_type.getShape(), in_etype);
    out.setType(new_type);
    return success();
  }
};

// to make compare inputs have the same min max
struct CompareCalibartion : public OpRewritePattern<top::CompareOp> {
  using OpRewritePattern<top::CompareOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::CompareOp op,
                                PatternRewriter &rewriter) const override {
    Value l = op.getLhs();
    Value r = op.getRhs();
    if (false == module::isCalibratedType(l) ||
        false == module::isCalibratedType(r)) {
      return failure();
    }
    auto stype = module::getStorageType(l);
    auto l_ctype = module::getCalibratedType(l);
    auto r_ctype = module::getCalibratedType(r);
    auto max = std::max(l_ctype.getMax(), r_ctype.getMax());
    auto min = std::min(l_ctype.getMin(), r_ctype.getMin());
    if (l_ctype.getMax() == r_ctype.getMax() &&
        l_ctype.getMin() == r_ctype.getMin()) {
      return failure();
    }
    auto new_ctype = quant::CalibratedQuantizedType::get(stype, min, max);
    auto new_ltype = RankedTensorType::get(module::getShape(l), new_ctype);
    auto new_rtype = RankedTensorType::get(module::getShape(r), new_ctype);
    l.setType(new_ltype);
    r.setType(new_rtype);
    return success();
  }
};

template <typename TyOp>
struct BackwardMutiInSingleOut : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: need to be more clever
    for (auto in : op.getInputs()) {
      if (!module::isCalibratedType(in)) {
        return failure();
      }
      if (in.hasOneUse()) {
        continue;
      }
      for (auto user : in.getUsers()) {
        if (!isa<top::MaxPoolOp>(user) && user != op.getOperation()) {
          return failure();
        }
      }
    }

    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }
    // checkout all inputs have the same sign
    auto in_0 = op.getInputs()[0];
    auto in_0_qtype = module::getCalibratedType(in_0);
    bool un_signed = in_0_qtype.getMin() >= 0;
    for (uint i = 1; i < op.getInputs().size(); i++) {
      auto qtype = module::getCalibratedType(op.getInputs()[i]);
      if (un_signed != (qtype.getMin() >= 0)) {
        if (isa<top::ConcatOp>(op))
          return failure();
      }
    }

    auto out_qtype = module::getCalibratedType(out);
    // checkout all input cali is the same
    bool same = true;
    for (uint i = 1; i < op.getInputs().size(); i++) {
      auto qtype = module::getCalibratedType(op.getInputs()[i]);
      if (qtype.getMin() != in_0_qtype.getMin() ||
          qtype.getMax() != in_0_qtype.getMax()) {
        same = false;
        break;
      }
    }
    if (same) {
      if (out_qtype.getMin() == in_0_qtype.getMin() &&
          out_qtype.getMax() == in_0_qtype.getMax()) {
        // do nothing
        return failure();
      }
      auto out_type = out.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(out_type.getShape(), in_0_qtype);
      out.setType(new_type);
      return success();
    }

    for (Value in : op.getInputs()) {
      auto in_type = in.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
      in.setType(new_type);
      Backward(in);
    }
    return success();
  }
};

struct SelectiveWhere : public OpRewritePattern<top::WhereOp> {
  using OpRewritePattern<top::WhereOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::WhereOp op,
                                PatternRewriter &rewriter) const override {
    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }

    float const_v = 0.0;
    bool const_signed = false;
    if (op.getYIsConst()) {
      float c = op.getYConstVal().convertToDouble();
      const_signed = c < 0.;
      const_v = std::abs(c);
    }
    if (op.getXIsConst()) {
      float c = op.getXConstVal().convertToDouble();
      const_signed |= c < 0.;
      const_v = std::max(std::abs(c), const_v);
    }

    auto out_qtype = module::getCalibratedType(out);
    // if output th is less than const(if exists), make it larger to include
    // const val
    bool out_to_constv = false;
    if (out_qtype.getMax() < const_v) {
      auto out_qtype = module::getCalibratedType(out);
      auto new_qtype = quant::CalibratedQuantizedType::get(
          out_qtype.getExpressedType(),
          (const_signed || out_qtype.getMin() < 0.) ? -const_v * 0.1 : 0.0f,
          const_v);
      auto new_type = RankedTensorType::get(
          out.getType().cast<RankedTensorType>().getShape(), new_qtype);
      out.setType(new_type);
      out_to_constv = true;
    }
    // but if where is set float, don't backward the th
    bool float_where = false;
    if (LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str()) != LoweringConfig::quantize_map.end()) {
      if (LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str())->second == module::Mode::F32 ||
          LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str())->second == module::Mode::F16)
        float_where = true;
    }

    // if input is not the same with out, set the input to follow output
    // don't backward to condition, and don't backward to input if output has been enlarged to const_v
    bool changed = false;
    if (!op.getXIsConst() && !out_to_constv && !float_where) {
      auto in = op.getTbrn();
      if (!module::isCalibratedType(in))
        return failure();
      if (module::getCalibratedType(in).getMin() != out_qtype.getMin() ||
          module::getCalibratedType(in).getMax() != out_qtype.getMax()) {
        auto in_qtype = module::getCalibratedType(in);
        auto new_qtype = quant::CalibratedQuantizedType::get(
            in_qtype.getExpressedType(), out_qtype.getMin(),
            out_qtype.getMax());
        auto new_type = RankedTensorType::get(
            in.getType().cast<RankedTensorType>().getShape(), new_qtype);
        in.setType(new_type);
        changed |= true;
      }
    }
    if (!op.getYIsConst() && !out_to_constv && !float_where) {
      auto in = op.getFbrn();
      if (!module::isCalibratedType(in))
        return failure();
      if (module::getCalibratedType(in).getMin() != out_qtype.getMin() ||
          module::getCalibratedType(in).getMax() != out_qtype.getMax()) {
        auto in_qtype = module::getCalibratedType(in);
        auto new_qtype = quant::CalibratedQuantizedType::get(
            in_qtype.getExpressedType(), out_qtype.getMin(),
            out_qtype.getMax());
        auto new_type = RankedTensorType::get(
            in.getType().cast<RankedTensorType>().getShape(), new_qtype);
        in.setType(new_type);
        changed |= true;
      }
    }
    if (changed)
      return success();
    else
      return failure();
  }
};

struct SelectiveMaskedFill : public OpRewritePattern<top::MaskedFillOp> {
  using OpRewritePattern<top::MaskedFillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(top::MaskedFillOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: need to be more clever
    for (auto in : op->getOperands()) {
      if (!module::isCalibratedType(in)) {
        return failure();
      }
      if (!in.hasOneUse()) {
        return failure();
      }
    }

    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }

    float const_v = 0.0;
    bool const_signed = false;
    float c = op.getConstVal().convertToDouble();
    const_signed = c < 0.;
    const_v = std::abs(c);

    auto out_qtype = module::getCalibratedType(out);
    // if output th is less than const(if exists), make it larger to include
    // const val
    bool out_to_constv = false;
    if (out_qtype.getMax() < const_v) {
      auto out_qtype = module::getCalibratedType(out);
      auto new_qtype = quant::CalibratedQuantizedType::get(
          out_qtype.getExpressedType(),
          (const_signed || out_qtype.getMin() < 0.) ? -const_v * 0.1 : 0.0f,
          const_v);
      auto new_type = RankedTensorType::get(
          out.getType().cast<RankedTensorType>().getShape(), new_qtype);
      out.setType(new_type);
      out_to_constv = true;
    }
    // if input is not the same with out, set the input to follow output
    // don't backward to condition
    // but if where is set float, don't backward the th
    bool float_mf = false;
    if (LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str()) != LoweringConfig::quantize_map.end()) {
      if (LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str())->second == module::Mode::F32 ||
          LoweringConfig::quantize_map.find(module::getName(op.getOperation()).str())->second == module::Mode::F16)
        float_mf = true;
    }

    bool changed = false;
    auto in = op.getOperand(1);
    if ((module::getCalibratedType(in).getMin() != out_qtype.getMin() ||
         module::getCalibratedType(in).getMax() != out_qtype.getMax()) &&
        !out_to_constv && !float_mf) {
      auto in_qtype = module::getCalibratedType(in);
      auto new_qtype = quant::CalibratedQuantizedType::get(
          in_qtype.getExpressedType(), out_qtype.getMin(), out_qtype.getMax());
      auto new_type = RankedTensorType::get(
          in.getType().cast<RankedTensorType>().getShape(), new_qtype);
      in.setType(new_type);
      changed |= true;
    }
    if (changed)
      return success();
    else
      return failure();
  }
};

struct CastInputCV18xxPattern : public OpRewritePattern<tpu::CastOp> {
  using OpRewritePattern<tpu::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tpu::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto setOpResultType = [](Value value, Type eltType) {
      auto shape = module::getShape(value);
      auto type = RankedTensorType::get(shape, eltType);
      value.setType(type);
    };

    auto prevOp = op->getOperand(0).getDefiningOp();
    if (isa<tpu::ReshapeOp>(prevOp)) {
      prevOp = prevOp->getOperand(0).getDefiningOp();
    }
    if (!isa<top::InputOp>(prevOp)) {
      return failure();
    }
    auto storage_type = module::getStorageType(op->getResult(0));
    if (storage_type.isIntOrIndex() &&
        storage_type.getIntOrFloatBitWidth() == 16) {
      // setOpResultType(prevOp->getOperand(0), storage_type);
      setOpResultType(prevOp->getResult(0), storage_type);
      setOpResultType(op->getOperand(0), storage_type);
      rewriter.replaceOp(op, {op->getOperand(0)});
      return success();
    }
    return failure();
  }
};

/**
 * @brief Try insert tile since shapes cannot merge to 4d in some case
*/
template <typename TyOp>
struct TryInsertTileBinaryPattern : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  bool can_be_merged(int64_t a1, int64_t a2, int64_t b1, int64_t b2) const {
    // case 0: both dims are same --- always true
    if (a1 == b1 && a2 == b2)
      return true;
    // case 1: only one dim is same --- only when another is 1 can be merged
    if ((a1 == b1 && a2 != b2 && a1 == 1) || (a1 != b1 && a2 == b2 && a2 == 1))
      return true;
    // case 2: both dims are not same --- only a or b broadcast can be merged
    if (a1 != b1 && a2 != b2 && (a1 == a2 || b1 == b2))
      return true;
    return false;
  }

  bool canMergeTo4D(const std::vector<int64_t> &ashape,
                    const std::vector<int64_t> &bshape, int shape_dim) const {
    if (shape_dim > 4) {
      int i = 0;
      while (i < shape_dim - 1) {
        if (can_be_merged(ashape[i], ashape[i + 1], bshape[i], bshape[i + 1])) {
          --shape_dim;
        } else {
          ++i;
        }
        if (shape_dim == 4)
          break;
      }
    }
    return shape_dim <= 4;
  }

  bool needBroadcast(const std::vector<int64_t> &shape1,
                    const std::vector<int64_t> &shape2) const {
    int dim1 = shape1.size();
    int dim2 = shape2.size();
    int maxDim = std::max(dim1, dim2);
    for (int i = 1; i <= maxDim; ++i) {
        int size1 = (dim1 - i >= 0) ? shape1[dim1 - i] : 1;
        int size2 = (dim2 - i >= 0) ? shape2[dim2 - i] : 1;
        if (size1 != size2 && (size1 != 1 || size2 != 1)) {
            return true;
        }
    }
    return false;
  }

  void try_insert_tile(TyOp &op, PatternRewriter &rewriter, int idx, int axis, int tile) const {
    Value opd = op.getOperand(idx);
    auto def_op = opd.getDefiningOp();
    auto input_shape = module::getShape(opd);
    auto newType =
        RankedTensorType::get(input_shape, module::getStorageType(opd));
    auto name = module::getName(opd).str();
    if (opd && !isa<ReturnOp>(def_op)) {
        name += "_" + module::getName(op.getOperation()).str();
    }
    name += "_tile";
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> weight_tile(input_shape.size(), 1);
    weight_tile[axis] = tile;
    attrs.emplace_back(
        rewriter.getNamedAttr("tile", rewriter.getI64ArrayAttr(weight_tile)));
    auto tileOp =
        rewriter.create<tpu::TileOp>(loc, newType, ValueRange{opd, module::getNoneOp(op)}, attrs);
    op->setOperand(idx, tileOp);
    std::vector<int64_t> output_shape = input_shape;
    output_shape[axis] = tile;
    module::setShape(tileOp.getOutput(), output_shape);
  }

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    int max_allow_dim_backend = 4;
    Value out = op.getOutput();
    if (isa<ReturnOp>(op))
      return failure();
    int opd_num = op.getNumOperands();
    if (opd_num != 2)
      return failure();

    Value opd1 = op.getOperand(0);
    Value opd2 = op.getOperand(1);
    const std::vector<int64_t> shape1 = module::getShape(opd1);
    const std::vector<int64_t> shape2 = module::getShape(opd2);
    int shape_dim = std::max(shape1.size(), shape2.size());
    if (needBroadcast(shape1, shape2) &&
        !canMergeTo4D(shape1, shape2, shape_dim)) {

      for (int i = 0; i < shape_dim - max_allow_dim_backend; ++i) {
        if (shape1[i] == shape2[i]) {
          continue;
        } else if (shape1[i] == 1) {
          try_insert_tile(op, rewriter, 0, i, shape2[i]);
        } else if (shape2[i] == 1) {
          try_insert_tile(op, rewriter, 1, i, shape1[i]);
        }
      }
      return success();
    }
    return failure();
  }
};

struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp(module_);
    LoweringConfig::isQuantized = false;
    auto mode_ = StringRef(mode).upper();
    auto mode = module::symbolizeMode(mode_);
    assert(mode.has_value());
    module::setMode(mode.value());
    if (weightFileName != "") {
      module::setWeightFileName(weightFileName);
    }

    LoweringConfig::doWinograd = doWinograd.hasValue() ? doWinograd.getValue() : false;
    init_qtable();

    if (module::isState(module::State::TOP_QUANTIZED)) {
      module::setAsymmetric(true);
      LoweringConfig::isQuantized = true;
    } else {
      LoweringConfig::isQuantized = false;
      module::setAsymmetric(isAsymmetric);
      calibration_process();
    }
    module::setLinearQuantMode(LinearQuantMode);

    if ((module::isBM1684XFamily() || module::isSG2260Family())
         && !LoweringConfig::isQuantized &&
        (module::getMode() == module::Mode::INT8 ||
         module::getMode() == module::Mode::UINT8)) {
      qtable_process();
      module::updateModuleTypes();
    }

    RewritePatternSet patterns(ctx_);

    // process shape related ops
    if (module::isBM1684XFamily() || module::isSG2260Family()) {
      bm1684x::populateTopShapeToTpuConversionPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      bm1684::populateTopShapeToTpuConversionPatterns(&patterns);
    }

    applyPatternsAndFoldGreedily(module_, std::move(patterns));

    patterns.clear();
    patterns.add<TryInsertTileBinaryPattern<top::AddOp>,
                 TryInsertTileBinaryPattern<top::SubOp>,
                 TryInsertTileBinaryPattern<top::MulOp>,
                 TryInsertTileBinaryPattern<top::MaxOp>,
                 TryInsertTileBinaryPattern<top::MinOp>,
                 TryInsertTileBinaryPattern<top::CompareOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));

    patterns.clear();
    if (module::isBM1684XFamily() || module::isSG2260Family()) {
      ConversionTarget target(*ctx_);
      ScfTypeConverter typeConverter;
      target.addLegalDialect<mlir::func::FuncDialect, top::TopDialect,
                             tpu::TpuDialect>();
      target.addIllegalOp<top::IfOp, top::LoopOp>();

      target.addDynamicallyLegalOp<mlir::func::CallOp>(
          [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });
      bm1684x::populateTopCfOpToTpuConversionPatterns(patterns, typeConverter,
                                                      ctx_);
      if (failed(applyPartialConversion(module_, target, std::move(patterns))))
        signalPassFailure();
      patterns.clear();
    }
    host2device_convert_process();

    // process other ops
    if (module::isBM1684XFamily() || module::isSG2260Family()) {
      bm1684x::populateTopToTpuConversionPatterns(&patterns);
    } else if (module::isBM1684Family()) {
      bm1684::populateTopToTpuConversionPatterns(&patterns);
    } else if (module::isCV18xx()) {
      cv18xx::populateTopToTpuConversionPatterns(&patterns);
    } else {
      llvm_unreachable("Not Implemented");
    }
    auto config = GreedyRewriteConfig();
    config.maxIterations = 1; // apply each pattern only once.
    applyPatternsAndFoldGreedily(module_, std::move(patterns), config);
    // adjust reshape
    patterns.clear();
    patterns.add<ForwardTypePattern<tpu::ReshapeOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    cast_process();
    relu_process();
    if (module::isCV18xx()) {
      patterns.clear();
      patterns.add<CastInputCV18xxPattern>(ctx_);
      applyPatternsAndFoldGreedily(module_, std::move(patterns));
    }
    module::updateModuleTypes();
    module::setState(module::State::TPU_LOWERED);
  }

protected:
  void calibration_process() {
    if (!module::isState(module::State::TOP_CALIBRATED)) {
      return;
    }
    // clang-format off
    RewritePatternSet patterns(ctx_);
    patterns.add<ForwardCalibartion<top::ReshapeOp>,
                 ForwardCalibartion<top::PermuteOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    // keep sign for some ops, keep sign before backward speading to check the sign consistency in backward
    // backend not support in out not the same sign
    patterns.clear();
    patterns.add<KeepSignPattern<top::AvgPoolOp>, KeepSignPattern<top::MaxPoolOp>, /*KeepAddSignPattern,*/
                 SetSubConstSignPattern>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<BackwardMutiInSingleOut<top::ConcatOp>,
                 BackwardMutiInSingleOut<top::MinOp>,
                 BackwardMutiInSingleOut<top::MaxOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>,
                 BackwardCalibartion<top::MaxPoolWithMaskOp>,
                 BackwardCalibartion<top::Depth2SpaceOp>,
                 //BackwardCalibartion<top::LeakyReluOp, true>,
                //  BackwardCalibartion<top::PReluOp>,
                 BackwardCalibartion<top::AbsOp>>(ctx_);
    if (!module::isCV18xx()) {
      // notice when it's dominated by negative value
      // and factor is very small it'll cause cumulative error
      patterns.add<BackwardCalibartion<top::PReluOp, true>>(ctx_);
      patterns.add<BackwardCalibartion<top::LeakyReluOp, true>>(ctx_);
    } else {
      patterns.add<BackwardCalibartion<top::LeakyReluOp, false>>(ctx_);
      // need consideration
      patterns.add<BackwardCalibartion<top::ScatterNDOp, false>>(ctx_);
    }
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<CompareCalibartion>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<SelectiveWhere,
		SelectiveMaskedFill>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<ForwardCalibartion<top::ReluOp>,
                 ForwardCalibartion<top::MaxPoolOp>,
                 ForwardCalibartion<top::MinConstOp>,
                 ForwardCalibartion<top::MaxConstOp>,
                 ForwardCalibartion<top::MaxPoolWithMaskOp>,
                 ForwardCalibartion<top::MaxUnpoolOp>,
                 ForwardCalibartion<top::ReshapeOp>,
                 ForwardCalibartion<top::UnsqueezeOp>,
                 ForwardCalibartion<top::SqueezeOp>,
                 ForwardCalibartion<top::SliceOp>,
                 ForwardCalibartion<top::TileOp>,
                 ForwardCalibartion<top::PadOp>,
                 ForwardCalibartion<top::PermuteOp>,
                 ForwardCalibartion<top::ReverseOp>,
                 ForwardCalibartion<top::UpsampleOp>,
                 ForwardCalibartion<top::LeakyReluOp>,
                //  ForwardCalibartion<top::PReluOp>,
                 ForwardCalibartion<top::AbsOp>,
                 ForwardMulConst,
                 ForwardArg
                >(ctx_);
    // clang-format on
    if (!module::isCV18xx()) {
      // notice it will cause cumulative error
      patterns.add<ForwardCalibartion<top::PReluOp>>(ctx_);
    } else {
      patterns.add<ForwardCalibartion<top::ReduceOp>>(ctx_);
    }
    if (module::isBM1684Family()) {
      // TODO: support asymmetric mode
      patterns.add<ForwardCalibartion<top::AvgPoolOp>>(ctx_);
    }
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    // keep sign for some ops
    // backend not support in out not the same sign
    patterns.clear();
    patterns.add<KeepSignPattern<top::AvgPoolOp>,
                 KeepSignPattern<top::MaxPoolOp>, /*KeepAddSignPattern,*/
                 SetSubConstSignPattern>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<SelectiveWhere, SelectiveMaskedFill>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
  }

  void host2device_convert_process() {
    // return types
    mainFunc_.walk([&](Operation *op) {
      if (!isa<ReturnOp>(op))
        return;
      for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
        try_insert_host2device(op, idx);
      }
    });
  }

  void relu_process() {
    Builder builder(ctx_);
    mainFunc_.walk([&](Operation *op) {
      if (module::isTpuOp(op)) {
        if (op->hasTrait<trait::SupportFuseRelu>() || isa<tpu::ReluOp>(op)) {
          if (module::isUniformQuantized(op->getResult(0)) ||
              module::isUniformQuantized(op->getOperand(0))) {
            op->setAttr("relu_limit", builder.getF64FloatAttr(-1.0));
          }
        }
      }
    });
  }

  void cast_process() {
    // return types
    auto retTypes = mainFunc_.getResultTypes();
    mainFunc_.walk([&](Operation *op) {
      bool is_tpu = module::isTpuOp(op);
      if (isa<tpu::YieldOp>(op)) {
        // do nothing
      } else if (auto in_op = dyn_cast<top::InputOp>(op)) {
        auto mode = TypeCastMode::DO_NOTHING;
        mlir::Type target_type;
        if (module::isCV18xx() == false) {
          target_type = type_verify_case_same(op, 0, mode);
        }
        if (mode != TypeCastMode::DO_NOTHING) {
          auto in = in_op.getInput();
          Value out = in_op.getOutput();
          auto out_type = out.getType();
          out.setType(in.getType());
          auto castOp = do_cast(out, target_type, mode);
          castOp.setType(out_type);
          out.replaceAllUsesExcept(castOp, castOp.getDefiningOp());
        }
      } else if (is_tpu || isa<ReturnOp>(op)) {
        for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
          auto opd = op->getOperand(idx);
          if (module::isWeight(opd) || module::isNone(opd)) {
            continue;
          }
          auto mode = TypeCastMode::DO_NOTHING;
          mlir::Type target_type;
          if (auto typeIf = dyn_cast<TypeInterface>(op)) {
            target_type = typeIf.type_verify(idx, mode);
          } else if (isa<ReturnOp>(op)) {
            auto stype = module::getStorageType(opd);
            if (module::isUniformQuantized(opd) || stype.isBF16() ||
                stype.isF16()) {
              target_type = type_verify_case_type(op, idx, retTypes[idx], mode);
            }
          } else {
            target_type = type_verify_case_same(op, idx, mode);
          }
          if (mode != TypeCastMode::DO_NOTHING) {
            auto castOp = do_cast(opd, target_type, mode, op);
            op->setOperand(idx, castOp);
          }
        }
      }
    });
  }


  int userCnt(Operation *op) {
    auto out = op->getResult(0);
    if (out.getUsers().empty())
      return 0;
    else {
      int cnt = 0;
      auto x = out.user_begin();
      while (x != out.user_end()) {
        cnt++;
        x = std::next(x);
      }
      return cnt;
    }
  }

  //SISO is single input single output, not counting weight and none and input/output
  bool isSISO(Operation *op) {
    int cnt = 0;
    for (auto in:op->getOperands()) {
      if (isa<top::InputOp>(in.getDefiningOp()) ||
          isa<top::WeightOp>(in.getDefiningOp()) ||
          isa<top::NoneOp>(in.getDefiningOp()))
        continue;
      else {
        if (cnt != 0)
          return false;
        else
          cnt ++;
      }
    }
    if (cnt == 0)
      return false;
    return (userCnt(op) == 1);
  }

  // return a set of end layernorms after ffn , the count would be the number of encoder ffn part
  void match_encoder_ffn(std::set<Operation*> &ffn) {
    mainFunc_.walk([&](Operation *op) {
      if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
        if (auto addop = dyn_cast<top::AddOp>(lnop.getInput().getDefiningOp())) {
          if (!addop.getOutput().hasOneUse())
            return;
          top::MatMulOp mmop = NULL;
          top::LayerNormOp lnop1 = NULL;
          for (auto in: addop.getOperands()) {
            if (isa<top::LayerNormOp>(in.getDefiningOp()))
              lnop1 = dyn_cast_or_null<top::LayerNormOp>(in.getDefiningOp());
            else if ( isa<top::MatMulOp>(in.getDefiningOp()))
              mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
            else
              return;
          }
          if (mmop == NULL || lnop1 == NULL || !isSISO(mmop.getOperation()))
            return;
          if (isa<top::GELUOp>(mmop.getInput().getDefiningOp())) {
            auto geluop = dyn_cast_or_null<top::GELUOp>(mmop.getInput().getDefiningOp());
            if (!isSISO(geluop.getOperation()))
              return;
            if (isa<top::MatMulOp>(geluop.getInput().getDefiningOp())) {
              auto mmop1 = dyn_cast_or_null<top::MatMulOp>(geluop.getInput().getDefiningOp());
              if (mmop1.getInput().getDefiningOp() != lnop1 || !isSISO(mmop1.getOperation())) {
                return;
              }
              if (!addop.getOutput().hasOneUse() || !mmop.getOutput().hasOneUse() || !geluop.getOutput().hasOneUse() || !mmop1.getOutput().hasOneUse())
                return;
              ffn.insert(lnop);
              return;
            }
          }
        }
      }
    });
  }

  // return a set of end layernorms after ffn , the count would be the number of encoder ffn part
  void match_mha(std::set<Operation*> &mha) {
    mainFunc_.walk([&](Operation *op) {
      if (auto lnop = dyn_cast<top::LayerNormOp>(op)) {
        if (auto addop = dyn_cast<top::AddOp>(lnop.getInput().getDefiningOp())) {
          if (!addop.getOutput().hasOneUse()) {
            return;
          }
          top::MatMulOp mmop = NULL;
          top::LayerNormOp top_lnop = NULL;

          top::ReshapeOp reshapeop = NULL;
          top::PermuteOp pmop = NULL;
          top::MatMulOp mmop1 = NULL;

          for (auto in:addop.getOperands()) {
            if (isa<top::MatMulOp>(in.getDefiningOp())) {
              mmop = dyn_cast_or_null<top::MatMulOp>(in.getDefiningOp());
            }
            else if (isa<top::LayerNormOp>(in.getDefiningOp())) {
              top_lnop = dyn_cast_or_null<top::LayerNormOp>(in.getDefiningOp());
            }
            else
              return;
          }
          if (mmop == NULL || top_lnop == NULL || !isSISO(mmop.getOperation()))
            return;
          if (isa<top::ReshapeOp>(mmop.getInput().getDefiningOp())) {
            reshapeop = dyn_cast_or_null<top::ReshapeOp>(mmop.getInput().getDefiningOp());
          }
          if (reshapeop == NULL || !isSISO(reshapeop.getOperation()))
            return;
          if (isa<top::PermuteOp>(reshapeop.getInput().getDefiningOp())) {
            pmop = dyn_cast_or_null<top::PermuteOp>(reshapeop.getInput().getDefiningOp());
          }
          if (pmop == NULL || !isSISO(pmop.getOperation()))
            return;
          if (isa<top::MatMulOp>(pmop.getInput().getDefiningOp())) {
            mmop1 = dyn_cast_or_null<top::MatMulOp>(pmop.getInput().getDefiningOp());
          }
          if (mmop1 == NULL)
            return;

          top::PermuteOp pmv = NULL;
          top::ReshapeOp rsv = NULL;
          top::SoftmaxOp sm = NULL;

          for (auto in:mmop1.getOperands()) {
            if (isa<top::PermuteOp>(in.getDefiningOp()))
              pmv = dyn_cast_or_null<top::PermuteOp>(in.getDefiningOp());
            else if (isa<top::SoftmaxOp>(in.getDefiningOp()))
              sm = dyn_cast_or_null<top::SoftmaxOp>(in.getDefiningOp());
            else if (isa<top::NoneOp>(in.getDefiningOp()))
              continue;
            else
              return;
          }
          if (pmv == NULL || sm == NULL)
            return;

          // check value branch
          if (isa<top::ReshapeOp>(pmv.getInput().getDefiningOp()))
            rsv = dyn_cast_or_null<top::ReshapeOp>(pmv.getInput().getDefiningOp());
          if (rsv == NULL || !isSISO(rsv.getOperation()))
            return;
          if (isa<top::MatMulOp>(rsv.getInput().getDefiningOp())) {
            auto mm_ = dyn_cast_or_null<top::MatMulOp>(rsv.getInput().getDefiningOp());
            if (mm_ == NULL || !isSISO(mm_.getOperation()) || mm_.getInput().getDefiningOp() != top_lnop)
              return;
          }

          // check k,v branches
          top::AddOp addop1 = NULL;
          top::MulConstOp mcop = NULL;
          if (isa<top::AddOp>(sm.getInput().getDefiningOp()))
            addop1 = dyn_cast_or_null<top::AddOp>(sm.getInput().getDefiningOp());
          if (!addop1 || !addop1.getOutput().hasOneUse())
            return;
          for (auto in:addop1.getOperands()) {
            top::MulConstOp mcop_ = NULL;
            if (isa<top::MulConstOp>(in.getDefiningOp()))
              mcop_ = dyn_cast_or_null<top::MulConstOp>(in.getDefiningOp());
            else
              return;
            if (mcop_.getConstVal().convertToDouble() == 0.125)
              mcop = mcop_;
            else if (mcop_.getConstVal().convertToDouble() == -10000)
              continue;
            else
              return;
          }
          if (mcop == NULL || !isSISO(mcop.getOperation()))
            return;

          top::MatMulOp mmop2 = NULL;
          if (isa<top::MatMulOp>(mcop.getInput().getDefiningOp()))
            mmop2 = dyn_cast_or_null<top::MatMulOp>(mcop.getInput().getDefiningOp());
          if (mmop2 == NULL)
            return;
          int inputs = 0;
          for (auto in : mmop2.getOperands()) {
            if (isa<top::WeightOp>(in.getDefiningOp()))
              continue;
            else if (isa<top::PermuteOp>(in.getDefiningOp())) {
              auto p_ = dyn_cast_or_null<top::PermuteOp>(in.getDefiningOp());
              if (p_ == NULL || !isSISO(p_.getOperation()))
                return;
              if (!isa<top::ReshapeOp>(p_.getInput().getDefiningOp()))
                return;
              auto r_ = dyn_cast_or_null<top::ReshapeOp>(p_.getInput().getDefiningOp());
              if (r_ == NULL || !isSISO(r_.getOperation()))
                return;
              if (!isa<top::MatMulOp>(r_.getInput().getDefiningOp()))
                return;
              auto m_ = dyn_cast_or_null<top::MatMulOp>(r_.getInput().getDefiningOp());
              if (m_ == NULL || !isSISO(m_.getOperation()))
                return;
              if (m_.getInput().getDefiningOp() != top_lnop)
                return;
              inputs ++;
            }
          }
          if (inputs != 2)
            return;
          mha.insert(lnop);
        }
      }
    });
  }

  void match_attention(std::set<Operation*> &attention) {
    mainFunc_.walk([&](Operation *op) {
      if (auto atop = dyn_cast<top::AttentionOp>(op)) {
        attention.insert(op);
      }
    });
  }

  bool bert_mix_precision() {
    std::set<Operation*> ffn;
    std::set<Operation*> mha;
    std::set<Operation*> attention;

    match_encoder_ffn(ffn);
    match_mha(mha);
    match_attention(attention);

    if (ffn.size() > 0 && (mha.size() > 0 || attention.size() > 0)) {
      // now to set
      // 1. all add before layernorm to f16
      // 2. last matmul of mha output to f16
      // 3. add before softmax to f32
      for (auto op:mha) {
        auto addop = dyn_cast_or_null<top::AddOp>(dyn_cast<top::LayerNormOp>(op).getInput().getDefiningOp());
        if (addop == NULL)
          return false;
        if (LoweringConfig::quantize_map.find(module::getName(addop.getOperation()).str()) == LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert( {module::getName(addop.getOperation()).str(), module::Mode::F16});
        }
      }
      for (auto op:ffn) {
        auto addop = dyn_cast_or_null<top::AddOp>(dyn_cast<top::LayerNormOp>(op).getInput().getDefiningOp());
        if (addop == NULL)
          return false;
        if (LoweringConfig::quantize_map.find(module::getName(addop.getOperation()).str()) == LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert( {module::getName(addop.getOperation()).str(), module::Mode::F16});
        }
        for (auto in:addop.getOperands()) {
          if (auto mmop = dyn_cast<top::MatMulOp>(in.getDefiningOp())) {
            if (LoweringConfig::quantize_map.find(module::getName(mmop.getOperation()).str()) == LoweringConfig::quantize_map.end()) {
                LoweringConfig::quantize_map.insert( {module::getName(mmop.getOperation()).str(), module::Mode::F16});
            }
          }
        }
      }

      return true;
    }
    else
      return false;
  }

  void set_add_before_softmax_fp32() {
    mainFunc_.walk([&](Operation *op) {
      if (auto addop = dyn_cast<top::AddOp>(op)) {
        if (isa<top::InputOp>(addop)) {
          return;
        }
        if (LoweringConfig::quantize_map.find(module::getName(op).str()) !=
            LoweringConfig::quantize_map.end())
          return;

        int idx = 0;
        float th[2] = {0.0};
        for (auto in : addop.getInputs()) {
          if (!module::isCalibratedType(in))
            return;
          if (isa<top::WeightOp>(in.getDefiningOp())) {
            auto weight =
                dyn_cast<top::WeightOp>(in.getDefiningOp()).read<float>();
            float absmax = fabs(weight.get()->at(0));
            for (int i = 0; i < weight.get()->size(); i++) {
              float value = fabs(weight.get()->at(i));
              absmax = value > absmax ? value : absmax;
            }
            th[idx++] = absmax;
          } else {
            auto in_type = module::getCalibratedType(in);
            th[idx++] = std::max(std::abs(in_type.getMin()), std::abs(in_type.getMax()));
          }
          if (idx > 2)
            return;
        }
        if (th[0] < 1e-8 || th[1] < 1e-8)
          return;
        if (th[0] / th[1] > 64 || th[1] / th[0] > 64) {
          if (LoweringConfig::quantize_map.find(
                  module::getName(op).str()) ==
              LoweringConfig::quantize_map.end()) {
            LoweringConfig::quantize_map.insert(
                {module::getName(op).str(), module::Mode::F32});
          }
        }
      }
    });
  }

  void qtable_process() {
    bert_mix_precision();
    set_add_before_softmax_fp32();
  }

  Value do_cast(Value v, Type to, TypeCastMode mode,
                Operation *user_op = nullptr) {
    auto to_stype = module::getStorageType(to);
    // check whether value has been casted
    for (auto user : v.getUsers()) {
      if (false == isa<tpu::CastOp>(user) &&
          (false == isa<tpu::GenericCpuOp>(user) ||
           dyn_cast<tpu::GenericCpuOp>(user).getCpuOpName() != "quant")) {
        continue;
      }
      if (type_need_cast(user->getResult(0).getType(), to) == false) {
        return user->getResult(0);
      }
    }

    bool all_next_layer_is_int4 = false;
    if (module::getMode() == module::Mode::INT4) {
      all_next_layer_is_int4 = true;
      for (auto user : v.getUsers()) {
        if (!isa<tpu::Conv2DOp, tpu::MatMulOp>(user)) {
          all_next_layer_is_int4 = false;
        } else if (isa<tpu::Conv2DOp>(user)) {
          auto conv = dyn_cast<tpu::Conv2DOp>(user);
          auto conv_attr = getConv2DParam(conv);
          if (conv_attr.is_dw /*|| conv_attr.sw > 1*/) {
            all_next_layer_is_int4 = false;
          }
        }
      }
    }

    auto ctx = v.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(v);
    auto name = module::getName(module::getOriValue(v)).str();
    if (user_op && !isa<ReturnOp>(user_op)) {
      name += module::getName(user_op).str();
    }
    switch (mode) {
    case TypeCastMode::DO_DEQUANTIZE:
    case TypeCastMode::DO_CAST: {
      name += "_" + type_string(to_stype);
      auto newType = RankedTensorType::get(module::getShape(v), to_stype);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      if (module::getOriValue(v).getDefiningOp()
           ->hasTrait<trait::ShapeProducer>()) {
        auto castOp =
            builder.create<tpu::ShapeCastOp>(loc, newType, ValueRange{v});
        return castOp.getOutput();
      } else {
        auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
        return castOp.getOutput();
      }
    }
    case TypeCastMode::DO_QUANTIZE: {
      if (module::isCalibratedType(v) == false) {
        v.dump();
        llvm_unreachable("Only calibrated type can do quantize");
      }
      auto newType = getQuantInt8Type(v, module::isAsymmetric());
      if (all_next_layer_is_int4) {
        newType = getQuantInt4Type(v, module::isAsymmetric());
      }
      name += "_" + type_string(newType);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      if (module::isCV18xx()) {
        auto parentOp = v.getDefiningOp();
        if (isa<top::InputOp>(parentOp)) {
          return insert_18xx_cpu_cast(builder, v, loc, newType);
        }
      }
      auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
      return castOp.getOutput();
    }
    default:
      break;
    }
    return v;
  }

  Value insert_18xx_cpu_cast(OpBuilder &builder, Value &v, NameLoc &loc,
                             Type &newType) {
    float scale = module::getUniformQuantizedType(newType).getScale();
    scale = 1 / scale;
    std::vector<NamedAttribute> attrs;
    std::vector<NamedAttribute> param;
    attrs.emplace_back(
        builder.getNamedAttr("cpu_op_name", builder.getStringAttr("quant")));
    param.emplace_back(
        builder.getNamedAttr("from", builder.getStringAttr("FP32")));
    param.emplace_back(
        builder.getNamedAttr("to", builder.getStringAttr("INT8")));
    param.emplace_back(
        builder.getNamedAttr("scale", builder.getF32FloatAttr(scale)));
    attrs.emplace_back(
        builder.getNamedAttr("param", builder.getDictionaryAttr(param)));
    auto castOp = builder.create<tpu::GenericCpuOp>(
        loc, newType, ValueRange{v}, ArrayRef<NamedAttribute>{attrs});
    return castOp.getOutputs()[0];
  }

  static module::Mode qmode(const std::string &mode) {
    std::string tmp = StringRef(mode).upper();
    auto mode_ = module::symbolizeMode(tmp);
    if (mode_.has_value()) {
      return mode_.value();
    }
    llvm::errs() << "Unknown quantize mode: [" << mode << "]\n";
    llvm_unreachable("Unknown quantize mode");
    return module::Mode::F32;
  }

  void init_qtable() {
    LoweringConfig::quantize_map.clear();
    if (ignore_f16_overflow == false &&
        module::getMode() == module::Mode::F16) {
      mainFunc_.walk([&](Operation *op) {
        // if have other op need convert from f16 to f32, add here
        // if need better performence, just set ignore_f16_overflow in model_deploy
        // defaultly we need ensure the computation is correct
        if (isa<top::LayerNormOp, top::RMSNormOp, top::AvgPoolOp>(op)) {
          auto name = module::getName(op).str();
          LoweringConfig::quantize_map[name] = module::Mode::F32;
        }
      });
    }
    if (qtable.empty()) {
      return;
    }
    std::regex map_pattern("\\S+\\s+\\S+");
    std::regex name_pattern("\\S+");
    std::regex info_pattern("#.*");
    std::regex empty_pattern("^\\s*$");
    std::ifstream infile(qtable);
    if (!infile) {
      llvm::errs() << "Can't open file: " << qtable << " !\n";
      llvm_unreachable("Open quantize table failed");
    }
    std::string line;
    while (std::getline(infile, line)) {
      if (line.back() == '\r') {
        line.pop_back();
      }
      std::istringstream iss(line);
      std::string name;
      std::string mode;
      if (std::regex_match(line, empty_pattern)) {
        continue;
      }
      if (std::regex_match(line, info_pattern)) {
        continue;
      }
      if (std::regex_match(line, map_pattern)) {
        iss >> name;
        iss >> mode;
        if (module::isCV18xx()) {
          if (StringRef(mode).upper() == "F32" ||
              StringRef(mode).upper() == "F16")
            mode = "BF16";
        }
        LoweringConfig::quantize_map[name] = qmode(mode);
        continue;
      }
      if (std::regex_match(line, name_pattern)) {
        continue;
      }

      llvm::errs() << "Error, quantize file in [" << line << "]\n";
      assert(false);
    }
  }

protected:
  ModuleOp module_;
  FuncOp mainFunc_;
  MLIRContext *ctx_;
};

std::unique_ptr<Pass> createConvertTopToTpu() {
  return std::make_unique<ConvertTopToTpu>();
}

} // namespace tpu_mlir
