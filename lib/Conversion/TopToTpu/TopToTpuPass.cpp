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
#include "tpu_mlir/Support/Module.h"
#include <fstream>
#include <regex>
#include <sstream>

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

namespace tpu_mlir {

template <typename OpTy> static void BackwardOp(OpTy op) {
  Value in = op.getInput();
  Value out = op.getOutput();
  auto in_type = in.getType().cast<RankedTensorType>();
  auto out_qtype = module::getCalibratedType(out);
  auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
  in.setType(new_type);
}

static void Backward(Value in) {
  if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
    BackwardOp(reshapeOp);
    // Backward(reshapeOp.getInput());
  } else if (auto permuteOp = dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
    BackwardOp(permuteOp);
    // Backward(permuteOp.getInput());
  }
}

template <typename OpTy> static void ForwardOp(OpTy op) {
  Value in = op.getInput();
  Value out = op.getOutput();
  auto out_type = out.getType().cast<RankedTensorType>();
  auto in_qtype = module::getCalibratedType(in);
  auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
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
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
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
    auto out_qtype = module::getCalibratedType(out);
    // checkout all input cali is the same
    auto in_0 = op.getInputs()[0];
    auto in_0_qtype = module::getCalibratedType(in_0);
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

struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = module::getMainFuncOp();
    LoweringConfig::isQuantized = false;
    auto mode_ = StringRef(mode).upper();
    auto mode = module::symbolizeMode(mode_);
    assert(mode.has_value());
    module::setMode(mode.value());
    if (weightFileName != "") {
      module::setWeightFileName(weightFileName);
    }
    if (module::isState(module::State::TOP_QUANTIZED)) {
      module::setAsymmetric(true);
      LoweringConfig::isQuantized = true;
    } else {
      LoweringConfig::isQuantized = false;
      module::setAsymmetric(isAsymmetric);
      calibration_process();
    }
    init_qtable();

    if (module::isBM1684XFamily() && is_bert_model() &&
        !LoweringConfig::isQuantized) {
      set_bert_mix_precision_process();
      module::updateModuleTypes();
    }

    RewritePatternSet patterns(ctx_);

    // process shape related ops
    if (module::isBM1684XFamily()) {
      bm1684x::populateTopShapeToTpuConversionPatterns(&patterns);
    }

    applyPatternsAndFoldGreedily(module_, std::move(patterns));

    patterns.clear();
    if (module::isBM1684XFamily()) {
      ConversionTarget target(*ctx_);
      ScfTypeConverter typeConverter;
      target.addLegalDialect<mlir::func::FuncDialect, top::TopDialect,
                             tpu::TpuDialect>();
      target.addIllegalOp<top::IfOp>();

      target.addDynamicallyLegalOp<mlir::func::CallOp>(
          [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });
      bm1684x::populateTopCfOpToTpuConversionPatterns(patterns, typeConverter,
                                                      ctx_);
      if (failed(applyPartialConversion(module_, target, std::move(patterns))))
        signalPassFailure();
      patterns.clear();
    }
    hd_convert_process();

    // process other ops
    if (module::isBM1684XFamily()) {
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
    patterns.clear();
    patterns.add<BackwardMutiInSingleOut<top::ConcatOp>,
                 BackwardMutiInSingleOut<top::MinOp>,
                 BackwardMutiInSingleOut<top::MaxOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>,
                 BackwardCalibartion<top::MaxPoolWithMaskOp>,
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
    }
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<CompareCalibartion>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    patterns.add<ForwardCalibartion<top::ReluOp>,
                 ForwardCalibartion<top::MaxPoolOp>,
                 ForwardCalibartion<top::MaxPoolWithMaskOp>,
                 ForwardCalibartion<top::MaxUnpoolOp>,
                 ForwardCalibartion<top::ReshapeOp>,
                 ForwardCalibartion<top::SliceOp>,
                 ForwardCalibartion<top::TileOp>,
                 ForwardCalibartion<top::PadOp>,
                 ForwardCalibartion<top::PermuteOp>,
                 ForwardCalibartion<top::ReverseOp>,
                 ForwardCalibartion<top::UpsampleOp>,
                 ForwardCalibartion<top::LeakyReluOp>,
                //  ForwardCalibartion<top::PReluOp>,
                 ForwardCalibartion<top::AbsOp>,
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
    patterns.add<KeepSignPattern<top::AvgPoolOp>, KeepAddSignPattern>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));

    // if (module::isBM1684XFamily()) {
    //   subconst_sign_process();
    //   mulconst_scale_process();
    //   module::updateModuleTypes();
    // }
  }

  void hd_convert_process() {
    // return types
    auto retTypes = mainFunc_.getResultTypes();
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
            auto castOp = do_cast(opd, target_type, mode);
            op->setOperand(idx, castOp);
          }
        }
      }
    });
  }

  bool is_bert_model() {
    auto mOp = getOperation();
    bool bert = false;
    auto getUserCnt = [](Operation *op) {
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
    };
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
          if (auto mulconstop = dyn_cast<top::MulConstOp>(op)) {
            if (getUserCnt(mulconstop) == 12 || getUserCnt(mulconstop) == 24) {
              for (auto nopmc : mulconstop->getResult(0).getUsers()) {
                if (auto addop = dyn_cast<top::AddOp>(nopmc)) {
                  if (getUserCnt(addop) != 1) {
                    return;
                  }
                  for (auto nopadd : addop->getResult(0).getUsers()) {
                    if (auto softmaxop = dyn_cast<top::SoftmaxOp>(nopadd)) {
                      continue;
                    } else {
                      return;
                    }
                  }
                } else {
                  return;
                }
              }
              bert = true;
            }
          }
        }
      });
    }
    return bert;
  }

  void subconst_sign_process() {
    auto mOp = getOperation();
    for (auto func : mOp.getOps<FuncOp>()) {
      if (module::isAsymmetric())
        return;
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
          if (auto subcop = dyn_cast<top::SubConstOp>(op)) {
            auto in = subcop.getInput();
            auto out = subcop.getOutput();
            auto in_stype = module::getStorageType(in);
            auto out_stype = module::getStorageType(out);
            auto in_ctype = module::getCalibratedType(in);
            auto out_ctype = module::getCalibratedType(out);
            auto in_min = in_ctype.getMin();
            auto out_min = out_ctype.getMin();
            auto out_max = out_ctype.getMax();
            if (in_min >= 0 && out_min <= 0) {
              if (module::getMode() ==
                  module::Mode::INT8) { // should check the op not the module
                auto new_out_type = quant::CalibratedQuantizedType::get(
                    out_stype, -out_max * 128.0 / 127.0, out_max);
                auto new_outr_type =
                    RankedTensorType::get(module::getShape(out), new_out_type);
                out.setType(new_outr_type);
              } else {
                llvm_unreachable("un-supported calibrated type for subconst!");
              }
            }
          }
        }
      });
    }
  }

  void mulconst_scale_process() {
    auto mOp = getOperation();
    for (auto func : mOp.getOps<FuncOp>()) {
      if (module::isAsymmetric())
        return;
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
          if (auto mulcop = dyn_cast<top::MulConstOp>(op)) {
            auto in = mulcop.getInput();
            auto out = mulcop.getOutput();
            auto out_stype = module::getStorageType(out);
            auto in_ctype = module::getCalibratedType(in);
            auto out_ctype = module::getCalibratedType(out);
            auto in_min = in_ctype.getMin();
            auto in_max = in_ctype.getMax();
            auto const_v = std::abs(mulcop.getConstVal().convertToDouble());
            if (module::getMode() ==
                module::Mode::INT8) { // should check the op not the module
              auto new_out_type = quant::CalibratedQuantizedType::get(
                  out_stype, in_min * const_v, in_max * const_v);
              auto new_outr_type =
                  RankedTensorType::get(module::getShape(out), new_out_type);
              out.setType(new_outr_type);
            } else {
              llvm_unreachable("un-supported calibrated type for subconst!");
            }
          }
        }
      });
    }
  }

  void set_bert_mix_precision_process() {
    auto mOp = getOperation();
    for (auto func : mOp.getOps<FuncOp>()) {
      if (module::isAsymmetric())
        return;
      func.walk([&](Operation *op) {
        if (op->getLoc().dyn_cast<NameLoc>() && !module::isOpInGroup(op)) {
          if (auto addop = dyn_cast<top::AddOp>(op)) {
            if (isa<top::InputOp>(addop)) {
              return;
            }
            int input_ok = 0;
            for (auto opd : addop.getOperands()) {
              if (auto layernormop =
                      dyn_cast<top::LayerNormOp>(opd.getDefiningOp())) {
                input_ok++;
              } else if (auto matmulop =
                             dyn_cast<top::MatMulOp>(opd.getDefiningOp())) {
                if (LoweringConfig::quantize_map.find(
                        module::getName(opd.getDefiningOp()).str()) ==
                    LoweringConfig::quantize_map.end()) {
                  LoweringConfig::quantize_map.insert(
                      {module::getName(opd.getDefiningOp()).str(),
                       module::Mode::F16});
                }
                input_ok++;
              } else {
                input_ok = 0;
                return;
              }
            }
            for (auto re : addop.getResult().getUsers()) {
              if (auto layernormop = dyn_cast<top::LayerNormOp>(re)) {
                if (input_ok == 2)
                  input_ok++;
              } else {
                input_ok = 0;
                return;
              }
            }

            if (LoweringConfig::quantize_map.find(module::getName(op).str()) ==
                LoweringConfig::quantize_map.end()) {
              LoweringConfig::quantize_map.insert(
                  {module::getName(op).str(), module::Mode::F16});
            }
          }
        }
      });
    }
  }

  Value do_cast(Value v, Type to, TypeCastMode mode) {
    auto from_stype = module::getStorageType(v);
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
    auto name = module::getName(v).str();
    switch (mode) {
    case TypeCastMode::DO_DEQUANTIZE:
    case TypeCastMode::DO_CAST: {
      name += "_" + type_string(to_stype);
      auto newType = RankedTensorType::get(module::getShape(v), to_stype);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      if (v.getDefiningOp()->hasTrait<trait::ShapeProducer>()) {
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
