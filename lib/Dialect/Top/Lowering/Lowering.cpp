//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Lowering.h"

#include <map>
using namespace tpu_mlir::trait;
namespace tpu_mlir {
namespace top {

Value do_transfer(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto in_shape = Module::getShape(in);
  auto out_type = Quant::getQuantInt8Type(out, asymmetric);
  auto ele_type = out_type.cast<RankedTensorType>().getElementType();
  auto new_type = RankedTensorType::get(in_shape, ele_type);

  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = Module::getName(in.getDefiningOp());
  auto out_name = Module::getName(op);
  auto new_name = in_name + "_to_" + out_name;
  int multiplier, rshift;
  get_scale_and_shift(in_scale / out_scale, multiplier, rshift, 8);
  if (in_zp == 0 && out_zp == 0) {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    auto in_type = in.getType().cast<RankedTensorType>();
    auto in_shape = in_type.getShape();
    builder.setInsertionPointAfterValue(in);
    auto mrOp = builder.create<tpu::MulShiftOp>(name_loc, new_type,
                                                ValueRange{in}, attrs);
    return mrOp.output();
  } else {
    std::vector<NamedAttribute> attrs;
    auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
    attrs.push_back(builder.getNamedAttr(
        "multiplier", builder.getSI32IntegerAttr(multiplier)));
    attrs.push_back(
        builder.getNamedAttr("rshift", builder.getI64IntegerAttr(rshift)));
    attrs.push_back(builder.getNamedAttr(
        "quant_mode",
        tpu::RequantModeAttr::get(op->getContext(), tpu::RequantMode::Normal)));
    builder.setInsertionPointAfterValue(in);
    auto rqOp = builder.create<tpu::RequantIntOp>(name_loc, new_type,
                                                  ValueRange{in}, attrs);
    return rqOp.output();
  }
}

Value do_transfer_fp(Value in, Value out, bool asymmetric) {
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, asymmetric);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, asymmetric);
  if (in_scale == out_scale && in_zp == out_zp) {
    return in;
  }
  auto op = out.getDefiningOp();
  OpBuilder builder(op);
  auto in_name = Module::getName(in.getDefiningOp());
  auto in_stype = Module::getStorageType(in);
  float offset = out_zp;
  auto in_shape = Module::getShape(in);
  auto rq_in = in;
  if (in_stype.isInteger(8) || in_zp != 0 && out_zp != 0) {
    auto add_name = in_name + "_add_zp";
    auto add_type = RankedTensorType::get(in_shape, builder.getI32Type());
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        builder.getNamedAttr("const_val", builder.getF64FloatAttr(in_zp)));
    auto name_loc = NameLoc::get(builder.getStringAttr(add_name));
    auto addOp = builder.create<tpu::AddConstOp>(name_loc, add_type,
                                                 ValueRange{in}, attrs);
    rq_in = addOp.output();
  } else if (in_zp != 0 && out_zp == 0) {
    offset = in_scale / out_scale * (-in_zp);
  }

  auto out_name = Module::getName(op);
  auto new_name = in_name + "_to_" + out_name;

  auto rq_stype = Module::getElementType(out);
  auto rq_type = RankedTensorType::get(in_shape, rq_stype);
  std::vector<NamedAttribute> attrs;
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  attrs.push_back(builder.getNamedAttr(
      "scale", builder.getF64FloatAttr(in_scale / out_scale)));
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getF64FloatAttr(offset)));
  attrs.push_back(builder.getNamedAttr(
      "quant_mode",
      tpu::RequantModeAttr::get(op->getContext(), tpu::RequantMode::Normal)));
  auto rqOp = builder.create<tpu::RequantFpOp>(name_loc, rq_type,
                                               ValueRange{rq_in}, attrs);
  if (out_zp == 0) {
    return rqOp.output();
  } else {
    llvm_unreachable("Not support now.\n");
  }
}

Value create_lookup_table(Value in, Value out, activate_f func,
                          bool asymmetric) {
  assert(func != nullptr);
  double in_scale, out_scale;
  int64_t in_zp, out_zp;
  bool in_sign, out_sign;
  Quant::getScaleAndZeroPoint(in, in_scale, in_zp, in_sign, asymmetric);
  Quant::getScaleAndZeroPoint(out, out_scale, out_zp, out_sign, asymmetric);
  int64_t min = in_sign ? -128 : 0;
  int64_t max = in_sign ? 127 : 255;
  auto op = out.getDefiningOp();
  OpBuilder builder(op->getContext());
  auto table_type = RankedTensorType::get({1, 1, 1, 256},
                                          builder.getIntegerType(8, out_sign));
  if (out_sign) {
    std::vector<int8_t> table(256, 0);
    for (auto i = min; i <= max; i++) {
      double data = (i - in_zp) * in_scale;
      data = func(data) / out_scale + out_zp;
      int index = i < 0 ? 256 + i : i;
      table[index] = Quant::to_int8(data);
    }
    return top::WeightOp::create(out.getDefiningOp(), "table", table,
                                 table_type);
  } else {
    std::vector<uint8_t> table(256, 0);
    for (auto i = min; i <= max; i++) {
      double data = (i - in_zp) * in_scale;
      data = func(data) / out_scale + out_zp;
      int index = i < 0 ? 256 + i : i;
      table[index] = Quant::to_uint8(data);
    }
    return top::WeightOp::create(out.getDefiningOp(), "table", table,
                                 table_type);
  }
}

Value create_lookup_table(Operation *owner, const std::vector<float> &table) {
  OpBuilder builder(owner->getContext());
  auto table_type = RankedTensorType::get({1, 1, 1, 256}, builder.getF32Type());
  return top::WeightOp::create(owner, "table", table, table_type);
}

Value do_dequant(Value input, Type to_type, int64_t multiplier, int64_t shift,
                 tpu::DequantMode mode, int64_t lshift) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(Module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("shift", builder.getI64IntegerAttr(shift)));
  if (mode == tpu::DequantMode::TFlite) {
    attrs.push_back(
        builder.getNamedAttr("lshift", builder.getI64IntegerAttr(lshift)));
  }
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::DequantModeAttr::get(ctx, mode)));

  std::string new_name =
      Module::getName(input.getDefiningOp()).str() + "_dequant";
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp = builder.create<tpu::DequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.output();
}

Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(Module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(builder.getNamedAttr("multiplier",
                                       builder.getSI32IntegerAttr(multiplier)));
  attrs.push_back(
      builder.getNamedAttr("rshift", builder.getI64IntegerAttr(-shift)));
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp = builder.create<tpu::RequantIntOp>(name_loc, newType,
                                                 ValueRange{input}, attrs);
  return newOp.output();
}

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode) {
  auto from_stype = Module::getStorageType(input);
  auto to_stype = Module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  std::vector<Value> operands = {input, quant};

  auto newType = to_type;
  if (tensorType == false) {
    newType = RankedTensorType::get(Module::getShape(input), to_stype);
  }

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("quant_mode", tpu::RequantModeAttr::get(ctx, mode)));

  auto newOp =
      builder.create<tpu::RequantIntAxisOp>(name_loc, newType, operands, attrs);
  return newOp.output();
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

struct LoweringPattern : public RewritePattern {
  LoweringPattern(MLIRContext *context, StringRef mode,
                  const std::map<Operation *, llvm::StringRef> &quantize_map)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context), mode(mode),
        quantize_map(quantize_map) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto lowering_op = dyn_cast<tpu_mlir::LoweringInterface>(op);
    if (!lowering_op) {
      return failure();
    }
    auto real_mode = mode;
    auto iter = quantize_map.find(op);
    if (iter != quantize_map.end()) {
      real_mode = iter->second;
    }
    if (real_mode == Quant::Type::INT8) {
      if (op->hasTrait<SupportFuseRelu>() || isa<top::ReluOp>(op)) {
        op->setAttr("relu_limit", rewriter.getF64FloatAttr(-1.0));
      }
    }
    auto module = Module::getModuleOp(op);
    auto chip = Module::getChip(module);
    if (chip == Module::Chip::BM1684) {
      if (real_mode == Quant::Type::F32) {
        lowering_op.lowering_f32_bm1684(rewriter);
      } else {
        lowering_op.lowering_int8_bm1684(rewriter);
      }
    } else if (chip == Module::Chip::BM1684x) {
      bool asymmetric = Module::getAsymmetric(module);
      if (Quant::isUniformQuantized(op->getResult(0))) {
        lowering_op.lowering_quant_bm1684x(rewriter);
      } else if (real_mode == Quant::Type::INT8) {
        lowering_op.lowering_int8_bm1684x(rewriter, asymmetric);
      } else if (real_mode == Quant::Type::F32) {
        lowering_op.lowering_f32_bm1684x(rewriter);
      } else if (real_mode == Quant::Type::BF16) {
        lowering_op.lowering_bf16_bm1684x(rewriter);
      } else if (real_mode == Quant::Type::F16) {
        lowering_op.lowering_f16_bm1684x(rewriter);
      } else {
        llvm_unreachable("unknown mode");
      }
    } else {
      llvm_unreachable("unknown chip");
    }
    return success();
  }

protected:
  StringRef mode;
  const std::map<Operation *, llvm::StringRef> &quantize_map;
};

class LoweringPass : public LoweringBase<LoweringPass> {
public:
  LoweringPass() {}

  void runOnOperation() override {
    module = getOperation();
    state_ = Module::getState(module);
    llvm::errs() << "default quantize mode:" << this->mode << ", is asymmetric "
                 << this->isAsymmetric << ", chip :" << this->chip
                 << ", state:" << state_ << "\n";
    chip_ = StringRef(chip).upper();
    Module::setChip(module, chip_);
    mode_ = StringRef(mode).upper();
    Module::setMode(module, mode_);
    ctx_ = module.getContext();
    mainFunc_ = Module::getMainFuncOp(module);
    if (Module::State::TOP_QUANTIZED == state_) {
      Module::setAsymmetric(module, true);
      asymmetric_ = true;
    } else {
      Module::setAsymmetric(module, isAsymmetric);
      asymmetric_ = isAsymmetric;
      calibration_process();
    }
    lowering_process();
    cast_process();
    Module::updateModuleTypes(module);
    Module::setState(module, Module::State::TPU_LOWERED);
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
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    patterns.clear();
    // clang-format off
    patterns.add<BackwardCalibartion<top::ReluOp>,
                 BackwardCalibartion<top::MaxPoolOp>,
                 BackwardCalibartion<top::ReshapeOp>,
                 BackwardCalibartion<top::LeakyReluOp>,
                 BackwardCalibartion<top::AbsOp>>(ctx_);
    // clang-format on
    applyPatternsAndFoldGreedily(module, std::move(patterns));
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
                 ForwardCalibartion<top::AbsOp>
                >(ctx_);
    // clang-format on
    if (chip_ == Module::Chip::BM1684) {
      // TODO: support asymmetric mode
      patterns.add<ForwardCalibartion<top::AvgPoolOp>>(ctx_);
    }
    applyPatternsAndFoldGreedily(module, std::move(patterns));
  }

  void lowering_process() {
    // lowering
    RewritePatternSet patterns(ctx_);
    patterns.add<LoweringPattern>(ctx_, mode_, quantize_map);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
    // adjust reshape
    patterns.clear();
    patterns.add<ForwardTypePattern<tpu::ReshapeOp>>(ctx_);
    applyPatternsAndFoldGreedily(module, std::move(patterns));
  }

  void cast_process() {
    // return types
    auto retTypes = mainFunc_.getResultTypes();
    mainFunc_.walk([&](Operation *op) {
      bool is_tpu = (op->getDialect()->getNamespace() == "tpu");
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
      auto newType = Quant::getQuantInt8Type(v, asymmetric_);
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

protected:
  ModuleOp module;
  FuncOp mainFunc_;
  llvm::StringRef state_;
  std::string chip_;
  std::string mode_;
  bool asymmetric_;
  std::map<Operation *, llvm::StringRef> quantize_map;
  MLIRContext *ctx_;
};

std::unique_ptr<OperationPass<ModuleOp>> createLoweringPass() {
  return std::make_unique<LoweringPass>();
}

} // namespace top
} // namespace tpu_mlir
