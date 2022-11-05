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

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOTPU
#include "tpu_mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace tpu_mlir {

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
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    auto in_qtype = Quant::getCalibratedType(in);
    auto out_qtype = Quant::getCalibratedType(out);
    if (in_qtype.getMax() == out_qtype.getMax() &&
        in_qtype.getMin() == out_qtype.getMin()) {
      return failure();
    }
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(out_type.getShape(), in_qtype);
    out.setType(new_type);
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
    if (!Quant::isCalibratedType(in)) {
      return failure();
    }
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    if (in.hasOneUse() == false) {
      return failure();
    }

    auto in_qtype = Quant::getCalibratedType(in);
    auto out_qtype = Quant::getCalibratedType(out);
    if (in_qtype.getMax() == out_qtype.getMax() &&
        in_qtype.getMin() == out_qtype.getMin()) {
      return failure();
    }
    auto in_type = in.getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
    in.setType(new_type);
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

    auto out = op.output();
    if (!Quant::isCalibratedType(out)) {
      return failure();
    }
    auto out_qtype = Quant::getCalibratedType(out);

    for (Value in : op.inputs()) {
      auto in_type = in.getType().cast<RankedTensorType>();
      auto new_type = RankedTensorType::get(in_type.getShape(), out_qtype);
      in.setType(new_type);
    }
    return success();
  }
};

struct ConvertTopToTpu : public ::impl::ConvertTopToTpuBase<ConvertTopToTpu> {
public:
  void runOnOperation() override {
    module_ = getOperation();
    ctx_ = &getContext();
    mainFunc_ = Module::getMainFuncOp(module_);
    state_ = Module::getState(module_);
    LoweringConfig::context = ctx_;
    LoweringConfig::chip = StringRef(chip).upper();
    LoweringConfig::mode = StringRef(mode).upper();
    LoweringConfig::isAsymmetric = isAsymmetric;
    Module::setChip(module_, LoweringConfig::chip);
    Module::setMode(module_, LoweringConfig::mode);

    if (Module::State::TOP_QUANTIZED == state_) {
      Module::setAsymmetric(module_, true);
      LoweringConfig::isAsymmetric = true;
    } else {
      Module::setAsymmetric(module_, LoweringConfig::isAsymmetric);
      if (Module::isCV18xx(LoweringConfig::chip)) {
        all_int8_process();
        Module::updateModuleTypes(module_);
      }
      calibration_process();
    }
    init_qtable();
    RewritePatternSet patterns(ctx_);
    ConversionTarget target(*ctx_);
    target.addLegalDialect<tpu::TpuDialect, func::FuncDialect>();
    // no need to lowering:
    target.addLegalOp<top::InputOp, top::WeightOp, top::NoneOp>();
    if (LoweringConfig::chip == Module::Chip::BM1684x) {
      bm1684x::populateTopToTpuConversionPatterns(&patterns);
    } else if (LoweringConfig::chip == Module::Chip::BM1684) {
      bm1684::populateTopToTpuConversionPatterns(&patterns);
    } else if (Module::isCV18xx(LoweringConfig::chip)) {
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
    Module::updateModuleTypes(module_);
    Module::setState(module_, Module::State::TPU_LOWERED);
  }

protected:
  void calibration_process() {
    if (state_ != Module::State::TOP_CALIBRATED) {
      return;
    }
    RewritePatternSet patterns(ctx_);
    patterns.add<BackwardMutiInSingleOut<top::ConcatOp>,
                 BackwardMutiInSingleOut<top::MinOp>,
                 BackwardMutiInSingleOut<top::MaxOp>>(ctx_);
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    // clang-format off
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>,
                 BackwardCalibartion<top::ReshapeOp>,
                 BackwardCalibartion<top::LeakyReluOp>,
                 BackwardCalibartion<top::PReluOp>,
                 BackwardCalibartion<top::AbsOp>>(ctx_);
    // clang-format on
    applyPatternsAndFoldGreedily(module_, std::move(patterns));
    patterns.clear();
    // clang-format off
    patterns.add<ForwardCalibartion<top::ReluOp>,
                 ForwardCalibartion<top::MaxPoolOp>,
                 ForwardCalibartion<top::SliceOp>,
                 ForwardCalibartion<top::TileOp>,
                 ForwardCalibartion<top::PadOp>,
                 ForwardCalibartion<top::ReshapeOp>,
                 ForwardCalibartion<top::PermuteOp>,
                 ForwardCalibartion<top::UpsampleOp>,
                 ForwardCalibartion<top::LeakyReluOp>,
                 ForwardCalibartion<top::PReluOp>,
                 ForwardCalibartion<top::AbsOp>
                >(ctx_);
    // clang-format on
    if (LoweringConfig::chip == Module::Chip::BM1684) {
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
          if (!Quant::isCalibratedType(value)) {
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
      if (false == isa<tpu::CastOp>(user)) {
        continue;
      }
      if (type_need_cast(user->getResult(0).getType(), to) == false) {
        return user->getResult(0);
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
      auto newType = Quant::getQuantInt8Type(v, LoweringConfig::isAsymmetric);
      name += "_" + type_string(newType);
      auto loc = NameLoc::get(builder.getStringAttr(name));
      auto castOp = builder.create<tpu::CastOp>(loc, newType, ValueRange{v});
      return castOp.output();
    }
    default:
      break;
    }
    return v;
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
  llvm::StringRef state_;
};

std::unique_ptr<Pass> createConvertTopToTpu() {
  return std::make_unique<ConvertTopToTpu>();
}

} // namespace tpu_mlir
