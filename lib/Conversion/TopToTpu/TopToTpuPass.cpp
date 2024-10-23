//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/ConvertTopToTpu.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "tpu_mlir/Support/ActiveUtils.h"
#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Backend/Arch.h"
#include <regex>

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

static void ForwardSign(Value out) {
  for (auto user : out.getUsers()) {
    if (auto avpOp = dyn_cast<top::AvgPoolOp>(user)) {
      ForwardOp(avpOp);
    } else if (auto mxpOp = dyn_cast<top::MaxPoolOp>(user)) {
      ForwardOp(mxpOp);
    } else if (auto absOp = dyn_cast<top::AbsOp>(user)) {
      ForwardOp(absOp);
    } else {
      Forward(out);
    }
  }
}

template <typename OpTy>
struct ForwardCalibartion : public OpRewriterPatternEx<OpTy> {
public:
  ForwardCalibartion(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"ForwardCalibartion") {}

  mlir::LogicalResult matchAndRewriteImpl(OpTy op,
                                          mlir::PatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<OpTy, top::ReduceOp>) {
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

  bool shouldPrint(OpTy opTy) const override { return false;}
};

struct ForwardMulConst : public OpRewriterPatternEx<top::MulConstOp> {
    ForwardMulConst(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::MulConstOp>(context,"ForwardMulConst") {}

  LogicalResult matchAndRewriteImpl(top::MulConstOp op,
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
  bool shouldPrint(top::MulConstOp op) const override { return false;}
};

struct ForwardArg : public OpRewriterPatternEx<top::ArgOp> {
  ForwardArg(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::ArgOp>(context,"ForwardArg") {}

  LogicalResult matchAndRewriteImpl(top::ArgOp op,
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
  bool shouldPrint(top::ArgOp op) const override { return false;}
};


template <typename OpTy>
struct KeepSignPattern : public OpRewriterPatternEx<OpTy> {
public:
  KeepSignPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"KeepSignPattern") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
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
    ForwardSign(out);
    return success();
  }
  bool shouldPrint(OpTy opTy) const override { return false;}
};


template <typename OpTy>
struct KeepMulSignPattern : public OpRewriterPatternEx<OpTy> {
public:
  KeepMulSignPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"KeepMulSignPattern") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto num_inputs = op.getInputs().size();
    if (num_inputs != 2)
      return failure();
    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }
    auto out_qtype = module::getCalibratedType(out);
    bool out_signed = out_qtype.getMin() < 0.0;
    bool in_signed[2] = {true, true};

    int idx = 0;
    for (auto i : op.getInputs()) {
      if (isa<top::WeightOp>(i.getDefiningOp())) {
        top::WeightOp w = dyn_cast<top::WeightOp>(i.getDefiningOp());
        auto filter_f32 = w.read<float>();
        if (filter_f32->size() != 1)
          return failure();
        if (filter_f32->at(0) >= 0.0)
          in_signed[idx] = false;
      } else {
        auto in_qtype = module::getCalibratedType(i);
        if (in_qtype.getMin() >= 0.0)
          in_signed[idx] = false;
      }
      idx++;
    }

    if (in_signed[0] == out_signed)
      return failure();
    else if (in_signed[1] == out_signed) {
      // switch inputs
      std::vector<Value> operands;
      for (auto in : op.getOperands()) {
        operands.insert(operands.begin(), in);
      }
      op.getOperation()->setOperands(operands);
      return success();
    } else {
      // two inputs are same but output is not the same
      if (in_signed[0]) {
        // in all signed, output unsigned, set output to signed, though possible
        // eg. sqr, but ic has the restriction
        float min = -out_qtype.getMax() * 0.1;
        auto etype = module::getStorageType(out);
        auto new_qtype =
            quant::CalibratedQuantizedType::get(etype, min, out_qtype.getMax());
        auto new_type = RankedTensorType::get(module::getShape(out), new_qtype);
        out.setType(new_type);
        Forward(out);
        return success();
      } else {
        // in all unsigned, output signed, may be caused by other pass? bad
        // cali_table?
        llvm_unreachable(
            (std::string("not reasonable, two unsigned get signed, check "
                         "cali-table and graph op is:") +
             std::string(module::getName(op.getOperation()).str()))
                .data());
      }
    }
    return failure();
  }
  bool shouldPrint(OpTy opTy) const override { return false;}
};


struct KeepAddSignPattern : public OpRewriterPatternEx<top::AddOp> {
public:
  KeepAddSignPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::AddOp>(context,"KeepAddSignPattern") {}

  LogicalResult matchAndRewriteImpl(top::AddOp op,
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
  bool shouldPrint(top::AddOp op) const override { return false;}
};


struct SetSubConstSignPattern : public OpRewriterPatternEx<top::SubConstOp> {
  public:

    SetSubConstSignPattern(mlir::MLIRContext *context)
        : OpRewriterPatternEx<top::SubConstOp>(context,"SetSubConstSignPattern") {}

  LogicalResult matchAndRewriteImpl(top::SubConstOp op,
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
  bool shouldPrint(top::SubConstOp op) const override { return false;}
};

struct SetSubSignPattern  : public OpRewriterPatternEx<top::SubOp> {
  public:
  SetSubSignPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::SubOp>(context,"SetSubSignPattern") {}

  LogicalResult matchAndRewriteImpl(top::SubOp op,
                                PatternRewriter &rewriter) const override {
    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }
    auto out_qtype = module::getCalibratedType(out);
    auto out_type = out.getType().cast<RankedTensorType>();
    if (out_qtype.getMin() >= 0) {
      auto new_out_type = quant::CalibratedQuantizedType::get(
          module::getStorageType(out), out_qtype.getMax() * (-0.1),
          out_qtype.getMax());
      auto new_type = RankedTensorType::get(out_type.getShape(), new_out_type);
      out.setType(new_type);
      Forward(out);
      return success();
    } else {
      return failure();
    }
  }
  bool shouldPrint(top::SubOp op) const override { return false;}
};

template <typename OpTy, bool KeepMin = false>
struct BackwardCalibartion : public OpRewriterPatternEx<OpTy> {
public:
  BackwardCalibartion(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"BackwardCalibartion") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
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
  bool shouldPrint(OpTy opTy) const override { return false;}
};

template <typename OpTy>
struct ForwardTypePattern : public OpRewriterPatternEx<OpTy> {
public:
 ForwardTypePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"ForwardTypePattern") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (module::isCV18xx()) {
      // for case input -> reshape -> anyOp
      //               |___anyOp
      // here should do quant manner otherwise will insert cast after shapeOp
      auto pre_op = op->getOperand(0).getDefiningOp();
      if (isa<top::InputOp>(pre_op))
        return failure();
    }
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
  bool shouldPrint(OpTy opTy) const override { return false;}
};


template <typename OpTy>
struct ForwardInt32TypePattern : public OpRewriterPatternEx<OpTy> {
public:
  ForwardInt32TypePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"ForwardInt32TypePattern") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
                                    PatternRewriter &rewriter) const override {
    auto pre_op = op->getOperand(0).getDefiningOp();
    if (isa<top::InputOp>(pre_op))
      return failure();
    Value in = op.getInput();
    Value out = op.getOutput();
    auto in_type = in.getType().cast<RankedTensorType>();
    auto out_type = out.getType().cast<RankedTensorType>();
    auto in_etype = in_type.getElementType();
    auto out_etype = out_type.getElementType();
    if (in_etype == out_etype || !in_etype.isInteger(32)) {
      return failure();
    }
    auto new_type = RankedTensorType::get(out_type.getShape(), in_etype);
    out.setType(new_type);
    return success();
  }
  bool shouldPrint(OpTy opTy) const override { return false;}
};

// to make compare inputs have the same min max
struct CompareCalibartion  : public OpRewriterPatternEx<top::CompareOp> {
  public:
  CompareCalibartion(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::CompareOp>(context,"CompareCalibartion") {}

  LogicalResult matchAndRewriteImpl(top::CompareOp op,
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
  bool shouldPrint(top::CompareOp op) const override { return false;}
};

template <typename OpTy>
struct  BackwardMutiInSingleOut : public OpRewriterPatternEx<OpTy> {
public:
   BackwardMutiInSingleOut(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"BackwardMutiInSingleOut") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
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
  bool shouldPrint(OpTy opTy) const override { return false;}
};


template <typename OpTy>
struct  BackwardAddThToMuls : public OpRewriterPatternEx<OpTy> {
public:
   BackwardAddThToMuls(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"BackwardAddThToMuls") {}

  LogicalResult matchAndRewriteImpl(OpTy op,
                                    PatternRewriter &rewriter) const override {
    // TODO: need to be more clever
    for (auto in : op.getInputs()) {
      if (!module::isCalibratedType(in)) {
        return failure();
      }
      if (!isa<top::MulOp>(in.getDefiningOp()))
        return failure();
      if (!in.hasOneUse()) {
        return failure();
      }
    }

    Value out = op.getOutput();
    if (!module::isCalibratedType(out)) {
      return failure();
    }

    auto out_qtype = module::getCalibratedType(out);

    for (Value in : op.getInputs()) {
      auto in_type = in.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
      in.setType(new_type);
      Backward(in);
    }
    return success();
  }
  bool shouldPrint(OpTy opTy) const override { return false;}
};


struct SelectiveWhere  : public OpRewriterPatternEx<top::WhereOp> {
  public:
  SelectiveWhere(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::WhereOp>(context,"SelectiveWhere") {}

  LogicalResult matchAndRewriteImpl(top::WhereOp op,
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
    if (LoweringConfig::quantize_map.find(
            module::getName(op.getOperation()).str()) !=
        LoweringConfig::quantize_map.end()) {
      if (LoweringConfig::quantize_map
                  .find(module::getName(op.getOperation()).str())
                  ->second == module::Mode::F32 ||
          LoweringConfig::quantize_map
                  .find(module::getName(op.getOperation()).str())
                  ->second == module::Mode::F16)
        float_where = true;
    }

    // if input is not the same with out, set the input to follow output
    // don't backward to condition, and don't backward to input if output has
    // been enlarged to const_v
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
  bool shouldPrint(top::WhereOp op) const override { return false;}
};

struct SelectiveMaskedFill : public OpRewriterPatternEx<top::MaskedFillOp> {
public:
  SelectiveMaskedFill(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::MaskedFillOp>(context,"SelectiveMaskedFill") {}

  LogicalResult matchAndRewriteImpl(top::MaskedFillOp op,
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
    if (LoweringConfig::quantize_map.find(
            module::getName(op.getOperation()).str()) !=
        LoweringConfig::quantize_map.end()) {
      if (LoweringConfig::quantize_map
                  .find(module::getName(op.getOperation()).str())
                  ->second == module::Mode::F32 ||
          LoweringConfig::quantize_map
                  .find(module::getName(op.getOperation()).str())
                  ->second == module::Mode::F16)
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
  bool shouldPrint(top::MaskedFillOp op) const override { return false;}
};

struct CastInputCV18xxPattern : public OpRewriterPatternEx<tpu::CastOp> {
public:
  CastInputCV18xxPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::CastOp>(context,"CastInputCV18xxPattern") {}

  LogicalResult matchAndRewriteImpl(tpu::CastOp op,
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
  bool shouldPrint(tpu::CastOp op) const override { return false;}
};

/**
 * @brief Try insert tile since shapes cannot merge to 4d in some case
 */

template <typename OpTy>
struct  TryInsertTileBinaryPattern : public OpRewriterPatternEx<OpTy> {
public:
   TryInsertTileBinaryPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context,"TryInsertTileBinaryPattern") {}

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

  static inline void merge_two_dims(std::vector<int64_t> &ashape,
                                    std::vector<int64_t> &bshape, int dims,
                                    int d_th) {
    ashape[d_th] *= ashape[d_th + 1];
    bshape[d_th] *= bshape[d_th + 1];
    for (int i = d_th + 1; i < dims - 1; i++) {
      ashape[i] = ashape[i + 1];
      bshape[i] = bshape[i + 1];
    }
  }

  bool canMergeTo4D(const std::vector<int64_t> &ashape,
                    const std::vector<int64_t> &bshape, int shape_dim) const {
    std::vector<int64_t> ashape_(8, 1);
    std::vector<int64_t> bshape_(8, 1);
    for (int i = 0; i < ashape.size(); i++) {
      ashape_[i] = ashape[i];
    }
    for (int i = 0; i < bshape.size(); i++) {
      bshape_[i] = bshape[i];
    }
    if (shape_dim > 4) {
      int i = 0;
      while (i < shape_dim - 1) {
        if (can_be_merged(ashape_[i], ashape_[i + 1], bshape_[i],
                          bshape_[i + 1])) {
          merge_two_dims(ashape_, bshape_, shape_dim, i);
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

  static void try_insert_tile(OpTy &op, PatternRewriter &rewriter, int idx,
                              int axis, int tile) {
    Value opd = op.getOperand(idx);
    auto def_op = opd.getDefiningOp();
    auto input_shape = module::getShape(opd);
    auto newType =
        RankedTensorType::get(input_shape, module::getElementType(opd));
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
        rewriter.create<top::TileOp>(loc, newType, ValueRange{opd}, attrs);
    op->setOperand(idx, tileOp);
    std::vector<int64_t> output_shape = input_shape;
    output_shape[axis] = tile;
    module::setShape(tileOp.getOutput(), output_shape);
  }

  LogicalResult matchAndRewriteImpl(OpTy op,
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

      for (int i = 0; i <= shape_dim - max_allow_dim_backend; ++i) {
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
    bool shouldPrint(OpTy opTy) const override { return false;}
};

struct TryInsertTileMatMulPattern  : public OpRewriterPatternEx<top::MatMulOp> {
public:
  TryInsertTileMatMulPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::MatMulOp>(context,"TryInsertTileMatMulPattern") {}

  LogicalResult matchAndRewriteImpl(top::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    Value opd1 = op.getOperand(0);
    Value opd2 = op.getOperand(1);
    const std::vector<int64_t> shape1 = module::getShape(opd1);
    const std::vector<int64_t> shape2 = module::getShape(opd2);
    if (shape1.size() <= 2 || shape2.size() <= 2)
      return failure();

    if (shape1.size() != shape2.size()) {
      return failure();
    }
    int shape_dim = shape1.size();
    int dims_merge_2_M = 0;
    for (int i = shape_dim - 3; i >= 0; i--) {
      if (shape2[i] == 1) {
        dims_merge_2_M++;
      } else {
        break;
      }
    }
    for (int i = shape_dim - 3 - dims_merge_2_M; i >= 0; --i) {
      if (shape1[i] == shape2[i])
        continue;
      else if (shape1[i] == 1) {
        TryInsertTileBinaryPattern<top::MatMulOp>::try_insert_tile(
            op, rewriter, 0, i, shape2[i]);
      } else if (shape2[i] == 1) {
        TryInsertTileBinaryPattern<top::MatMulOp>::try_insert_tile(
            op, rewriter, 1, i, shape1[i]);
      }
    }
    return failure();
  }
  bool shouldPrint(top::MatMulOp op) const override { return false;}
};

// prepare for W4A16 MatMul
struct W4A16MatMulPreparePattern  : public OpRewriterPatternEx<top::MatMulOp> {
  public:
  W4A16MatMulPreparePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<top::MatMulOp>(context,"W4A16MatMulPreparePattern") {}

  LogicalResult matchAndRewriteImpl(top::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto qmode = getOpQuantMode(op);
    if (!module::isWeight(op.getRight())) {
      return failure();
    }
    // W8A16
    if (qmode == module::Mode::W8BF16 || qmode == module::Mode::W8F16) {
      if (op.getWeightBits() == 8) {
        return failure();
      }
      op.setWeightBits(8);
      return success();
    }
    // W4A16
    if (qmode != module::Mode::W4BF16 && qmode != module::Mode::W4F16) {
      return failure();
    }
    if (op.getWeightBits() == 4) {
      return failure();
    }
    op.setWeightBits(4);
    if (module::getQuantGroupSize() <= 0)
      return success();
    auto out = op.getOutput();
    auto o_shape = module::getShape(out);
    auto o_name = module::getName(out);
    auto target_mode = (qmode == module::Mode::W4BF16 ? module::Mode::BF16
                                                      : module::Mode::F16);
    // if has q_group_size, means npu_num must be divided exactly by N
    auto r_shape = module::getShape(op.getRight());
    auto N = r_shape.back();
    if (N % backend::Arch::NPU_NUM == 0)
      return success();
    if (N < backend::Arch::NPU_NUM) {
      LoweringConfig::quantize_map[o_name.str()] = target_mode;
      return success();
    }
    if (!module::isWeight(op.getBias()) && !module::isNone(op.getBias())) {
      UNREACHABLE_OP("op filter is weight, but bias is not weight", op);
    }
    auto num_core = module::getCoreNum();
    auto N1 = N % (backend::Arch::NPU_NUM * num_core);
    auto N0 = N - N1;
    rewriter.setInsertionPoint(op);
    // op 0
    std::vector<Value> opds0;
    auto w0 = module::opSliceAxis(rewriter, op.getRight(), -1, 0, N0);
    opds0.push_back(op.getInput());
    opds0.push_back(w0);
    if (module::isWeight(op.getBias())) {
      auto b0 = module::opSliceAxis(rewriter, op.getBias(), -1, 0, N0);
      opds0.push_back(b0);
    } else {
      opds0.push_back(op.getBias());
    }
    auto loc0 = module::getLocLike(out, "0");
    std::vector<int64_t> shape0 = o_shape;
    shape0[o_shape.size() - 1] = N0;
    auto type0 = module::getTypeLike(out, shape0);
    auto m0_op =
        rewriter.create<top::MatMulOp>(loc0, type0, opds0, op->getAttrs());
    auto name0 = module::getName(m0_op.getOutput());
    LoweringConfig::quantize_map[name0.str()] = qmode;
    // op 1
    std::vector<Value> opds1;
    auto w1 = module::opSliceAxis(rewriter, op.getRight(), -1, N0, N1);
    opds1.push_back(op.getInput());
    opds1.push_back(w1);
    if (module::isWeight(op.getBias())) {
      auto b1 = module::opSliceAxis(rewriter, op.getBias(), -1, N0, N1);
      opds1.push_back(b1);
    } else {
      opds1.push_back(op.getBias());
    }
    auto loc1 = module::getLocLike(out, "1");
    std::vector<int64_t> shape1 = o_shape;
    shape1[o_shape.size() - 1] = N1;
    auto type1 = module::getTypeLike(out, shape1);
    auto m1_op =
        rewriter.create<top::MatMulOp>(loc1, type1, opds1, op->getAttrs());
    m1_op.removeWeightBitsAttr();
    auto name1 = module::getName(m1_op.getOutput());
    LoweringConfig::quantize_map[name1.str()] = target_mode;

    // concat this two op
    std::vector<Value> concat_operands = {m0_op.getOutput(), m1_op.getOutput()};
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr(
        "axis", rewriter.getSI32IntegerAttr(o_shape.size() - 1)));
    rewriter.replaceOpWithNewOp<top::ConcatOp>(op, op.getType(),
                                               concat_operands, attrs);
    LoweringConfig::quantize_map[o_name.str()] = target_mode;
    return success();
  }
  bool shouldPrint(top::MatMulOp op) const override { return false;}
};

// cast(u8->fp32) + active -> lut(u8->fp32)
// cast(u8->fp32) + active(fp32) + cast(fp32->fp16) -> lut(u8->fp16)
struct CastActivePattern : public OpRewriterPatternEx<tpu::ActiveOp> {
public:
  CastActivePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<tpu::ActiveOp>(context,"CastActivePattern") {}

  LogicalResult matchAndRewriteImpl(tpu::ActiveOp op,
                                PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (!isa<tpu::CastOp>(in_op) || !in_op->hasOneUse()) {
      return failure();
    }
    auto in = dyn_cast<tpu::CastOp>(in_op).getInput();
    auto out = op.getOutput();
    auto storage_itype = module::getStorageType(in);
    if (!storage_itype.isInteger(8) || !module::isUniformQuantized(in)) {
      return failure();
    }
    auto storage_type = module::getStorageType(out);
    if (!storage_type.isF32() && !storage_type.isF16() &&
        !storage_type.isBF16()) {
      return failure();
    }
    auto ctx = in.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(in);
    auto op_to_repl = op.getOperation();
    // if (op->hasOneUse()) {
    //   if (auto cast_op2 = dyn_cast<tpu::CastOp>(*out.getUsers().begin())) {
    //     auto out2 = cast_op2.getOutput();
    //     auto storage_type = module::getStorageType(out2);
    //     if (storage_type.isF16() || storage_type.isBF16()) {
    //       out = out2;
    //       op_to_repl = cast_op2.getOperation();
    //     }
    //   }
    // }
    auto table = create_lookup_table_fp(in, out, getActivateFunc(op));
    rewriter.replaceOpWithNewOp<tpu::LutOp>(op_to_repl, out.getType(),
                                            ValueRange{in, table});
    return success();
  }
  bool shouldPrint(tpu::ActiveOp op) const override { return false;}
};

void ConvertTopToTpu::runOnOperation() {
  module_ = getOperation();
  ctx_ = &getContext();
  mainFunc_ = module::getMainFuncOp(module_);
  LoweringConfig::isQuantized = false;
  module::setQuantGroupSize(quantGroupSize);
  if (weightFileName != "") {
    module::setWeightFileName(weightFileName);
  }

  RewritePatternSet patterns(ctx_);
  patterns.clear();
  patterns.add<TryInsertTileBinaryPattern<top::SubOp>,
               TryInsertTileBinaryPattern<top::MaxOp>,
               TryInsertTileBinaryPattern<top::MinOp>,
               TryInsertTileBinaryPattern<top::CompareOp>,
               TryInsertTileMatMulPattern>(ctx_);
  if (!module::isBM1684XFamily()) {
    patterns.add<TryInsertTileBinaryPattern<top::AddOp>,
                 TryInsertTileBinaryPattern<top::MulOp>>(ctx_);
  }
  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  patterns.clear();
  LoweringConfig::doWinograd =
      doWinograd.hasValue() ? doWinograd.getValue() : false;
  init_qtable();

  if (module::isState(module::State::TOP_QUANTIZED)) {
    module::setAsymmetric(true);
    LoweringConfig::isQuantized = true;
  } else {
    LoweringConfig::isQuantized = false;
    module::setAsymmetric(isAsymmetric);
    calibration_process();
  }

  if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
      !LoweringConfig::isQuantized &&
      (module::getMode() == module::Mode::INT8 ||
       module::getMode() == module::Mode::UINT8)) {
    // qtable_process();
    module::updateModuleTypes();
  }

  if ((module::isBM1684X() || module::isBM1688()) &&
      !LoweringConfig::isQuantized &&
      (module::getMode() == module::Mode::INT8 ||
       module::getMode() == module::Mode::UINT8)) {
    // handle matmul perchannel setting
    if (matmulPerchannel) {
      mainFunc_.walk([&](Operation *op) {
        if (isa<top::WeightOp, top::NoneOp, top::InputOp, ModuleOp, FuncOp,
                ReturnOp>(op)) {
          return;
        }
        if (isa<top::MatMulOp>(op)) {
          mlir::Attribute tmp = mlir::BoolAttr::get(op->getContext(), true);
          op->setAttr("matmulPerchannelQuant", tmp);
        }
      });
    }
  }
  if (module::isMARS3()) {
    mainFunc_.walk([&](Operation *op) {
      if (isa<top::WeightOp, top::NoneOp, top::InputOp, ModuleOp, FuncOp,
              ReturnOp>(op)) {
        return;
      }
      if (auto geluOp = dyn_cast<top::GELUOp>(op)) {
        if (geluOp.getApproxMode() == "normal")
          geluOp.setApproxMode(geluMode);
      }
    });
  }
  // process W4A16 MatMul
  if (!module::isState(module::State::TOP_QUANTIZED)) {
    module::applyPatternOnce<W4A16MatMulPreparePattern>(module_);
  }

  // kv_cache
  if ((module::isBM1684XFamily() || module::isBM1690Family()) &&
      (module::getMode() == module::Mode::W8F16 ||
       module::getMode() == module::Mode::W4F16 ||
       module::getMode() == module::Mode::W8BF16 ||
       module::getMode() == module::Mode::W4BF16 ||
       module::getMode() == module::Mode::F16) &&
       module::isState(module::State::TOP_CALIBRATED)) { // if calibration table presents
    kv_cache_process();
  }

  // process shape related ops
  if (module::isBM1684XFamily() || module::isBM1690Family()) {
    bm1684x::populateTopShapeToTpuConversionPatterns(&patterns);
  } else if (module::isBM1684Family()) {
    bm1684::populateTopShapeToTpuConversionPatterns(&patterns);
  }

  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  device2host_process();
  patterns.clear();
  if (module::isBM1684XFamily() || module::isBM1690Family()) {
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
  if (module::isBM1684XFamily() || module::isBM1690Family()) {
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
  patterns.add<
      ForwardTypePattern<tpu::ReshapeOp>, ForwardTypePattern<tpu::UnsqueezeOp>, ForwardTypePattern<tpu::SqueezeOp>,
      ForwardTypePattern<tpu::TileOp>, ForwardInt32TypePattern<tpu::SqueezeOp>,
      ForwardInt32TypePattern<tpu::SliceOp>, ForwardInt32TypePattern<tpu::PermuteOp>, ForwardInt32TypePattern<tpu::ShapeReduceOp>>(ctx_);
  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  cast_process();
  if (module::isBM1684XFamily()) {
    patterns.clear();
    patterns.add<CastActivePattern>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
  }
  relu_process();
  if (module::isCV18xx()) {
    patterns.clear();
    patterns.add<CastInputCV18xxPattern>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
  }
  module::updateModuleTypes();
  module::setState(module::State::TPU_LOWERED);
  bool hasTopOp = false;
  mainFunc_.walk([&](Operation *op) {
    if (isa<top::WeightOp, top::NoneOp, top::InputOp, ModuleOp, FuncOp,
            ReturnOp>(op)) {
      return;
    }
    if (!isa<tpu::TpuDialect>(op->getDialect())) {
      op->dump();
      hasTopOp = true;
    }
  });
  if (hasTopOp) {
    llvm_unreachable("unimplemented tpu dialect!");
  }
}

void ConvertTopToTpu::calibration_process() {
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
                 KeepSignPattern<top::AbsOp>,
                 SetSubConstSignPattern>(ctx_);

    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    if (!module::isCV18xx() && !module::isF8Modes()) {
      patterns.add<KeepMulSignPattern<top::MulOp>, /*KeepMulSignPattern,*/
                 SetSubConstSignPattern, SetSubSignPattern>(ctx_);
      applyPatternsAndFoldGreedily(module_, std::move(patterns));
      patterns.clear();
    }
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

    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      patterns.add<BackwardAddThToMuls<top::AddOp>>(ctx_);
      applyPatternsAndFoldGreedily(module_, std::move(patterns));
      patterns.clear();
    }
    patterns.add<CompareCalibartion>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    if (!module::isF8Modes()) {
      patterns.add<SelectiveWhere,
      SelectiveMaskedFill>(ctx_);
      applyPatternsAndFoldGreedily(module_, std::move(patterns));
      patterns.clear();
    }
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
               KeepSignPattern<top::AbsOp>, SetSubConstSignPattern>(ctx_);
  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  patterns.clear();
  patterns.add<SelectiveWhere, SelectiveMaskedFill>(ctx_);
  applyPatternsAndFoldGreedily(module_, std::move(patterns));
  patterns.clear();
}

void ConvertTopToTpu::host2device_convert_process() {
  // return types
  mainFunc_.walk([&](Operation *op) {
    if (!isa<ReturnOp>(op))
      return;
    for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
      try_insert_host2device(op, idx);
    }
  });
}

void ConvertTopToTpu::device2host_process() {
  mainFunc_.walk([&](Operation *op) {
    if (!op->hasTrait<trait::ShapeProducer>())
      return;
    if (isa<tpu::ShapeOp, tpu::Device2HostOp>(op))
      return;
    for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
      if (module::isNone(op->getOperand(idx)))
        continue;
      try_insert_device2host(op, idx);
    }
  });
}

void ConvertTopToTpu::relu_process() {
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

void ConvertTopToTpu::cast_process() {
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
        auto defByWeight = [&](auto &&m, Operation *op) {
          if (!op)
            return false;
          if (isa<top::WeightOp>(op))
            return true;
          else if (!isa<top::ReshapeOp, tpu::ReshapeOp>(op))
            return false;
          else
            return m(m, op->getOperand(0).getDefiningOp());
        };
        if (module::isWeight(opd) || module::isNone(opd) ||
            defByWeight(defByWeight, opd.getDefiningOp())) {
          continue;
        }
        auto inner_requant = op->getAttr("quant_inner_requant");
        if (inner_requant)
          continue;
        auto mode = TypeCastMode::DO_NOTHING;
        mlir::Type target_type;
        if (auto typeIf = dyn_cast<TypeInterface>(op)) {
          target_type = typeIf.type_verify(idx, mode);
        } else if (isa<ReturnOp>(op)) {
          auto stype = module::getStorageType(opd);
          if (module::isUniformQuantized(opd) || stype.isBF16() ||
              stype.isF16() || stype.isFloat8E4M3FN() || stype.isFloat8E5M2() ||
              (stype.isF32() && module::isCalibratedType(opd))) {
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

void ConvertTopToTpu::set_add_before_softmax_fp32() {
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
          th[idx++] =
              std::max(std::abs(in_type.getMin()), std::abs(in_type.getMax()));
        }
        if (idx > 2)
          return;
      }
      if (th[0] < 1e-8 || th[1] < 1e-8)
        return;
      if (th[0] / th[1] > 64 || th[1] / th[0] > 64) {
        if (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(op).str(), module::Mode::F16});
        }
      }
    }
  });
}

// void ConvertTopToTpu::qtable_process() {
//   bert_mix_precision();
//   swin_mix_precision();
//   cswin_mix_precision();
//   vit_mix_precision();
//   deit_mix_precision();
//   eva2_mix_precision();
//   detr_mix_precision();
//   set_add_before_softmax_fp32();
// }

// match kv cache
void ConvertTopToTpu::match_kv_cache(std::vector<Operation *> &kv_cache) {
  mainFunc_.walk([&](Operation *op) {
    if (auto addop = dyn_cast<top::AddOp>(op)) {
      top::MulOp mulop = NULL;
      top::ConcatOp ccop = NULL;
      if (isa<ReturnOp>(*(addop.getResult().getUsers().begin()))) {
        for (auto in : addop.getOperands()) {
          if (isa<top::MulOp>(in.getDefiningOp())) {
            mulop = dyn_cast_or_null<top::MulOp>(in.getDefiningOp());
          }
        }
        for (auto user : addop.getResult().getUsers()) {
          if (isa<top::ConcatOp>(user)) {
            ccop = dyn_cast_or_null<top::ConcatOp>(user);
          }
        }
      }
      if (mulop == NULL || ccop == NULL)
        return;
      else
        kv_cache.push_back(addop);
    }
    if (auto reshapeop = dyn_cast<top::ReshapeOp>(op)) {
      top::MatMulOp mmop = NULL;
      top::ConcatOp ccop = NULL;
      top::RMSNormOp rmsop = NULL;
      if (isa<ReturnOp>(*(reshapeop.getResult().getUsers().begin()))) {
        auto preOp = reshapeop->getOperands()[0].getDefiningOp();
        if (isa<top::MatMulOp>(preOp)) {
          mmop = dyn_cast_or_null<top::MatMulOp>(preOp);
          auto premmop = mmop->getOperands()[0].getDefiningOp();
          if (isa<top::RMSNormOp>(premmop)) {
            rmsop = dyn_cast_or_null<top::RMSNormOp>(premmop);
          }
        }
        for (auto user : reshapeop.getResult().getUsers()) {
          if (isa<top::ConcatOp>(user)) {
            ccop = dyn_cast_or_null<top::ConcatOp>(user);
          }
        }
      }
      if (mmop == NULL || rmsop == NULL || ccop == NULL)
        return;
      else
        kv_cache.push_back(reshapeop);
    }
  });
}

bool ConvertTopToTpu::kv_cache_mix_precision() {
  std::vector<Operation *> kv_cache;
  match_kv_cache(kv_cache);
  for (auto it = kv_cache.begin(); it != kv_cache.end(); ++it) {
    if (auto addop = dyn_cast<top::AddOp>(*it)) {
      if (LoweringConfig::quantize_map.find(
              module::getName(addop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(addop.getOperation()).str(), module::Mode::INT8});
      }
    }
    if (auto rsop = dyn_cast<top::ReshapeOp>(*it)) {
      if (LoweringConfig::quantize_map.find(
              module::getName(rsop.getOperation()).str()) ==
          LoweringConfig::quantize_map.end()) {
        LoweringConfig::quantize_map.insert(
            {module::getName(rsop.getOperation()).str(), module::Mode::INT8});
      }
      auto pre_reshape = rsop->getOperands()[0].getDefiningOp();
      if (isa<top::MatMulOp>(pre_reshape)) {
        auto pre_reshapeOp = dyn_cast<top::MatMulOp>(pre_reshape);
        if (LoweringConfig::quantize_map.find(
                module::getName(pre_reshapeOp.getOperation()).str()) ==
            LoweringConfig::quantize_map.end()) {
          LoweringConfig::quantize_map.insert(
              {module::getName(pre_reshapeOp.getOperation()).str(), module::Mode::INT8});
        }
      }
    }
  }
  return false;
}

void ConvertTopToTpu::kv_cache_process() {
  kv_cache_mix_precision();
}

Value ConvertTopToTpu::do_cast(Value v, Type to, TypeCastMode mode,
                               Operation *user_op) {
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
      } else if (isa<tpu::Conv2DOp>(user) || isa<tpu::MatMulOp>(user)) {
        if (isa<tpu::Conv2DOp>(user)) {
          auto conv = dyn_cast<tpu::Conv2DOp>(user);
          auto conv_attr = getConv2DParam(conv);
          if (conv_attr.is_dw /*|| conv_attr.sw > 1*/) {
            all_next_layer_is_int4 = false;
          }
        }
        auto user_name = module::getName(user).str();
        if (LoweringConfig::quantize_map.find(user_name) !=
            LoweringConfig::quantize_map.end()) {
          if (LoweringConfig::quantize_map[user_name] != module::Mode::INT4) {
            all_next_layer_is_int4 = false;
          }
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
    if (module::getOriValue(v)
            .getDefiningOp()
            ->hasTrait<trait::ShapeProducer>()) {
      auto castOp =
          builder.create<tpu::ShapeCastOp>(loc, newType, ValueRange{v});
      return castOp.getOutput();
    } else {
      if (module::getStorageType(v).isFloat8E4M3FN()) {
        name += std::string("_dequant");
        auto loc = NameLoc::get(builder.getStringAttr(name));
        double const_v =
            module::getCalibratedType(v).getMax() / get_f8e4m3_max();
        std::vector<NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr("const_val",
                                             builder.getF64FloatAttr(const_v)));
        auto mulOp =
            builder.create<tpu::MulConstOp>(loc, newType, ValueRange{v}, attrs);
        v.replaceAllUsesExcept(mulOp.getOutput(), mulOp);
        return mulOp.getOutput();
      } else {
        auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
        return castOp.getOutput();
      }
    }
  }
  case TypeCastMode::DO_QUANTIZE: {
    if (module::isCalibratedType(v) == false) {
      v.dump();
      llvm_unreachable("Only calibrated type can do quantize");
    }
    if (to.isFloat8E4M3FN()) {
      // auto newType = RankedTensorType::get(module::getShape(v),
      // module::getCalibratedType(v));
      builder.setInsertionPointAfterValue(v);
      name += std::string("_requant");
      float scale = get_f8e4m3_max() / module::getCalibratedType(v).getMax();
      auto value = do_requantFp(v, scale, 0.0, getQuantF8E4M3Type(v), name,
                                tpu::RequantMode::OnlyScale);
      return value;
    } else if (to.isFloat8E5M2()) {
      auto value = do_cast(v, getQuantF8E5M2Type(v), TypeCastMode::DO_CAST);
      return value;
    } else {
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
  }
  default:
    break;
  }
  return v;
}

Value ConvertTopToTpu::insert_18xx_cpu_cast(OpBuilder &builder, Value &v,
                                            NameLoc &loc, Type &newType) {
  float scale = module::getUniformQuantizedType(newType).getScale();
  scale = 1 / scale;
  std::vector<NamedAttribute> attrs;
  std::vector<NamedAttribute> param;
  attrs.emplace_back(
      builder.getNamedAttr("cpu_op_name", builder.getStringAttr("quant")));
  param.emplace_back(
      builder.getNamedAttr("from", builder.getStringAttr("FP32")));
  param.emplace_back(builder.getNamedAttr("to", builder.getStringAttr("INT8")));
  param.emplace_back(
      builder.getNamedAttr("scale", builder.getF32FloatAttr(scale)));
  attrs.emplace_back(
      builder.getNamedAttr("param", builder.getDictionaryAttr(param)));
  auto castOp = builder.create<tpu::GenericCpuOp>(
      loc, newType, ValueRange{v}, ArrayRef<NamedAttribute>{attrs});
  return castOp.getOutputs()[0];
}

module::Mode ConvertTopToTpu::qmode(const std::string &mode) {
  std::string tmp = StringRef(mode).upper();
  auto mode_ = module::symbolizeMode(tmp);
  if (mode_.has_value()) {
    return mode_.value();
  }
  llvm::errs() << "Unknown quantize mode: [" << mode << "]\n";
  llvm_unreachable("Unknown quantize mode");
  return module::Mode::F32;
}

void ConvertTopToTpu::init_qtable() {
  LoweringConfig::quantize_map.clear();
  if (ignore_f16_overflow == false && module::isF16Modes()) {
    mainFunc_.walk([&](Operation *op) {
      // if have other op need convert from f16 to f32, add here.
      // if need better performence, just set ignore_f16_overflow in
      // model_deploy. defaultly we need ensure the computation is correct.
      if (isa<top::AvgPoolOp>(op)) {
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
      auto src_mode = StringRef(mode).upper();
      if (module::isCV18xx()) {
        if (src_mode == "F32" || src_mode == "F16")
          mode = "BF16";
      }
      if ((src_mode == "W8F16" || src_mode == "W4F16") &&
          module::isBF16Modes()) {
        llvm_unreachable(
            "WxF16 and BF16 mix precision is not allowed, check your qtable");
      }
      if ((src_mode == "W8BF16" || src_mode == "W4BF16") &&
          module::isF16Modes()) {
        llvm_unreachable(
            "WxBF16 and F16 mix precision is not allowed, check your qtable");
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

std::unique_ptr<Pass> createConvertTopToTpu() {
  return std::make_unique<ConvertTopToTpu>();
}

} // namespace tpu_mlir
