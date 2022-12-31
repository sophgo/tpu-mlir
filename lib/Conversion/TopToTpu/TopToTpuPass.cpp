//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringBM1684X.h"
#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"
#include "tpu_mlir/Conversion/TopToTpu/TopToTpu.h"
#include <fstream>
#include <regex>
#include <sstream>
#include "tpu_mlir/Support/Helper/Module.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {

static void BackwardReshape(top::ReshapeOp op) {
  auto in = op.input();
  auto out = op.output();
  auto in_type = in.getType().cast<RankedTensorType>();
  auto out_qtype = Quant::getCalibratedType(out);
  auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
  in.setType(new_type);
  if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
    BackwardReshape(reshapeOp);
  }
}

static void ForwardReshape(top::ReshapeOp op) {
  auto in = op.input();
  auto out = op.output();
  auto out_type = out.getType().cast<RankedTensorType>();
  auto in_qtype = Quant::getCalibratedType(in);
  auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
  out.setType(new_type);
  if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
    ForwardReshape(reshapeOp);
  }
}

static void BackwardPermute(top::PermuteOp op) {
  auto in = op.input();
  auto out = op.output();
  auto in_type = in.getType().cast<RankedTensorType>();
  auto out_qtype = Quant::getCalibratedType(out);
  auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
  in.setType(new_type);
  if (auto permuteOp = dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
    BackwardPermute(permuteOp);
  }
}

static void ForwardPermute(top::PermuteOp op) {
  auto in = op.input();
  auto out = op.output();
  auto out_type = out.getType().cast<RankedTensorType>();
  auto in_qtype = Quant::getCalibratedType(in);
  auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
  out.setType(new_type);
  if (auto permuteOp = dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
    ForwardPermute(permuteOp);
  }
}

template <typename TyOp>
struct ForwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input();
    Value out = op.output();
    if (!Quant::isCalibratedType(in)) {
      return failure();
    }
    auto in_qtype = Quant::getCalibratedType(in);
    if (Quant::isCalibratedType(out)) {
      auto out_qtype = Quant::getCalibratedType(out);
      if (in_qtype.getMax() == out_qtype.getMax() &&
          in_qtype.getMin() == out_qtype.getMin()) {
        return failure();
      }
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
    if (auto reshapeOp = dyn_cast<top::ReshapeOp>(out.getDefiningOp())) {
      ForwardReshape(reshapeOp);
    } else if (auto permuteOp = dyn_cast<top::PermuteOp>(out.getDefiningOp())) {
      ForwardPermute(permuteOp);
    }
    return success();
  }
};

template <typename TyOp>
struct BackwardCalibartion : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op->getOperand(0);
    Value out = op.output();
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    if (in.hasOneUse() == false) {
      return failure();
    }

    auto out_qtype = Quant::getCalibratedType(out);
    if (Quant::isCalibratedType(in)) {
      auto in_qtype = Quant::getCalibratedType(in);
      if (in_qtype.getMax() == out_qtype.getMax() &&
          in_qtype.getMin() == out_qtype.getMin()) {
        return failure();
      }
    }
    auto in_type = in.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
    in.setType(new_type);
    if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
      BackwardReshape(reshapeOp);
    } else if (auto permuteOp = dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
      BackwardPermute(permuteOp);
    }
    return success();
  }
};

template <typename TyOp>
struct ForwardTypePattern : public OpRewritePattern<TyOp> {
  using OpRewritePattern<TyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TyOp op,
                                PatternRewriter &rewriter) const override {
    Value in = op.input();
    Value out = op.output();
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
    Value l = op.lhs();
    Value r = op.rhs();
    if (false == Quant::isCalibratedType(l) ||
        false == Quant::isCalibratedType(r)) {
      return failure();
    }
    auto stype = Module::getStorageType(l);
    auto l_ctype = Quant::getCalibratedType(l);
    auto r_ctype = Quant::getCalibratedType(r);
    auto max = std::max(l_ctype.getMax(), r_ctype.getMax());
    auto min = std::min(l_ctype.getMin(), r_ctype.getMin());
    if (l_ctype.getMax() == r_ctype.getMax() &&
        l_ctype.getMin() == r_ctype.getMin()) {
      return failure();
    }
    auto new_ctype = quant::CalibratedQuantizedType::get(stype, min, max);
    auto new_ltype = RankedTensorType::get(Module::getShape(l), new_ctype);
    auto new_rtype = RankedTensorType::get(Module::getShape(r), new_ctype);
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
    for (auto in : op.inputs()) {
      if (!Quant::isCalibratedType(in)) {
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

    Value out = op.output();
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    auto out_qtype = Quant::getCalibratedType(out);
    // checkout all input cali is the same
    auto in_0 = op.inputs()[0];
    auto in_0_qtype = Quant::getCalibratedType(in_0);
    bool same = true;
    for (uint i = 1; i < op.inputs().size(); i++) {
      auto qtype = Quant::getCalibratedType(op.inputs()[i]);
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

    for (Value in : op.inputs()) {
      auto in_type = in.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
      in.setType(new_type);
      if (auto reshapeOp = dyn_cast<top::ReshapeOp>(in.getDefiningOp())) {
        BackwardReshape(reshapeOp);
      } else if (auto permuteOp =
                     dyn_cast<top::PermuteOp>(in.getDefiningOp())) {
        BackwardPermute(permuteOp);
      }
    }
    return success();
  }
};

struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = Module::getMainFuncOp();
    LoweringConfig::isQuantized = false;
    Module::setChip(StringRef(chip).upper());
    Module::setMode(StringRef(mode).upper());

    if (Module::isState(Module::State::TOP_QUANTIZED)) {
      Module::setAsymmetric(true);
      LoweringConfig::isQuantized = true;
    } else {
      LoweringConfig::isQuantized = false;
      Module::setAsymmetric(isAsymmetric);
      if (Module::isCV18xx()) {
        all_int8_process();
        Module::updateModuleTypes();
      }
      calibration_process();
    }
    init_qtable();
    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<tpu::TpuDialect, func::FuncDialect>();
    // no need to lowering:
    target.addLegalOp<top::InputOp, top::WeightOp, top::NoneOp>();
    if (Module::isBM1684XFamily()) {
      bm1684x::populateTopToTpuConversionPatterns(&patterns);
    } else if (Module::isBM1684Family()) {
      bm1684::populateTopToTpuConversionPatterns(&patterns);
    } else if (Module::isCV18xx()) {
      cv18xx::populateTopToTpuConversionPatterns(&patterns);
    } else {
      llvm_unreachable("Not Implemented");
    }
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    // if (failed(
    //         applyPartialConversion(module_, target, std::move(patterns)))) {
    //   signalPassFailure();
    // }
    // adjust reshape
    patterns.clear();
    patterns.add<ForwardTypePattern<tpu::ReshapeOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    cast_process();
    relu_process();
    Module::updateModuleTypes();
    Module::setState(Module::State::TPU_LOWERED);
  }

protected:
  void calibration_process() {
    if (!Module::isState(Module::State::TOP_CALIBRATED)) {
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
                 // notice when it's dominated by negative value
                 // and factor is very small it'll cause cumulative error
                 BackwardCalibartion<top::LeakyReluOp>,
                //  BackwardCalibartion<top::PReluOp>,
                 BackwardCalibartion<top::AbsOp>>(ctx_);
    if (!Module::isCV18xx()) {
      // notice it will cause cumulative error
      patterns.add<BackwardCalibartion<top::PReluOp>>(ctx_);
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
                 // same issue as backward
                 ForwardCalibartion<top::LeakyReluOp>,
                //  ForwardCalibartion<top::PReluOp>,
                 ForwardCalibartion<top::AbsOp>
                >(ctx_);
    // clang-format on
    if (!Module::isCV18xx()) {
      // notice it will cause cumulative error
      patterns.add<ForwardCalibartion<top::PReluOp>>(ctx_);
    }
    if (Module::isBM1684Family()) {
      // TODO: support asymmetric mode
      patterns.add<ForwardCalibartion<top::AvgPoolOp>>(ctx_);
    }
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
  }

  void all_int8_process() {
    auto retTypes = mainFunc_.getResultTypes();
    mainFunc_.walk([&](Operation *op) {
      if (isa<tpu_mlir::InferenceInterface>(op) || isa<top::InputOp>(op)) {
        for (auto value : op->getResults()) {
          if (value.getType().isa<mlir::NoneType>() ||
              !Quant::isCalibratedType(value)) {
            continue;
          }
          auto out_qtype = Quant::getCalibratedType(value);
          if (out_qtype.getMin() != -out_qtype.getMax()) {
            auto max = out_qtype.getMax();
            auto quant_type = quant::CalibratedQuantizedType::get(
                out_qtype.getExpressedType(), -max, max);
            auto new_type =
                RankedTensorType::get(Module::getShape(value), quant_type);
            value.setType(new_type);
          }
        }
      }
    });
  }

  void relu_process() {
    Builder builder(ctx_);
    mainFunc_.walk([&](Operation *op) {
      if (Module::isTpuOp(op)) {
        if (op->hasTrait<trait::SupportFuseRelu>() || isa<tpu::ReluOp>(op)) {
          if (Quant::isUniformQuantized(op->getResult(0)) ||
              Quant::isUniformQuantized(op->getOperand(0))) {
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
      bool is_tpu = Module::isTpuOp(op);
      if (is_tpu || isa<func::ReturnOp>(op)) {
        for (uint32_t idx = 0; idx < op->getNumOperands(); idx++) {
          auto opd = op->getOperand(idx);
          TypeCastMode mode = TypeCastMode::DO_NOTHING;
          mlir::Type target_type;
          if (auto typeIf = dyn_cast<TypeInterface>(op)) {
            target_type = typeIf.type_verify(idx, mode);
          } else if (isa<func::ReturnOp>(op)) {
            // return op
            target_type = type_verify_case_type(op, idx, retTypes[idx], mode);
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

  Value do_cast(Value v, Type to, TypeCastMode mode) {
    auto from_stype = Module::getStorageType(v);
    auto to_stype = Module::getStorageType(to);
    // check whether value has been casted
    for (auto user : v.getUsers()) {
      if (false == isa<tpu::CastOp>(user) &&
          (false == isa<tpu::GenericCpuOp>(user) ||
           dyn_cast<tpu::GenericCpuOp>(user).operation_name() != "quant")) {
        continue;
      }
      if (type_need_cast(user->getResult(0).getType(), to) == false) {
        return user->getResult(0);
      }
    }

    bool all_next_layer_is_int4 = false;
    if (Module::getMode() == Quant::Type::INT4) {
      all_next_layer_is_int4 = true;
      for (auto user : v.getUsers()) {
        if (!isa<tpu::Conv2DOp, tpu::MatMulOp>(user)) {
          all_next_layer_is_int4 = false;
        }
      }
    }

    auto ctx = v.getContext();
    OpBuilder builder(ctx);
    builder.setInsertionPointAfterValue(v);
    auto name = Module::getName(v).str();
    switch (mode) {
    case TypeCastMode::DO_DEQUANTIZE:
    case TypeCastMode::DO_CAST: {
      name += "_" + type_string(to_stype);
      auto newType = RankedTensorType::get(Module::getShape(v), to_stype);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
      return castOp.output();
    }
    case TypeCastMode::DO_QUANTIZE: {
      if (Quant::isCalibratedType(v) == false) {
        v.dump();
        llvm_unreachable("Only calibrated type can do quantize");
      }
      auto newType = getQuantInt8Type(v, Module::isAsymmetric());
      if (all_next_layer_is_int4) {
        newType = getQuantInt4Type(v, Module::isAsymmetric());
      }
      name += "_" + type_string(newType);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      if (Module::isCV18xx()) {
        auto parentOp = v.getDefiningOp();
        if (isa<top::InputOp>(parentOp)) {
          return insert_18xx_cpu_cast(builder, v, loc, newType);
        }
      }
      auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
      return castOp.output();
    }
    default:
      break;
    }
    return v;
  }

  Value insert_18xx_cpu_cast(OpBuilder &builder, Value &v, NameLoc &loc,
                             Type &newType) {
    std::vector<NamedAttribute> attrs;
    std::vector<NamedAttribute> param;
    attrs.emplace_back(
        builder.getNamedAttr("operation_name", builder.getStringAttr("quant")));
    param.emplace_back(
        builder.getNamedAttr("from", builder.getStringAttr("FP32")));
    param.emplace_back(
        builder.getNamedAttr("to", builder.getStringAttr("INT8")));
    param.emplace_back(builder.getNamedAttr(
        "scale", builder.getF64FloatAttr(
                     1. / Quant::getUniformQuantizedType(newType).getScale())));
    attrs.emplace_back(
        builder.getNamedAttr("param", builder.getDictionaryAttr(param)));
    auto castOp = builder.create<tpu::GenericCpuOp>(
        loc, newType, ValueRange{v}, ArrayRef<NamedAttribute>{attrs});
    return castOp.output();
  }

  static StringRef qmode(const std::string &mode) {
    std::string tmp = StringRef(mode).upper();
    if (tmp == Quant::Type::INT8) {
      return Quant::Type::INT8;
    }
    if (tmp == Quant::Type::F16) {
      return Quant::Type::F16;
    }
    if (tmp == Quant::Type::BF16) {
      return Quant::Type::BF16;
    }
    if (tmp == Quant::Type::F32) {
      return Quant::Type::F32;
    }
    llvm::errs() << "Unknown quantize mode: [" << mode << "]\n";
    llvm_unreachable("Unknown quantize mode");
    return "";
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
