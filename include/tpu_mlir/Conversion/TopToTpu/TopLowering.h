//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
// ================================
// Lowering Helper Apis
// ================================

mlir::Type getQuantInt16Type(Value v, bool asymmetric = false);
mlir::Type getQuantInt8Type(Value v, bool asymmetric = false);
mlir::Type getQuantIntType(Value v, double scale, double offset, int bits = 8);
mlir::Type getQuantInt4Type(Value v, bool asymmetric = false);
mlir::Type getQuantBoolType(Value v);

template <typename ElemTy> static mlir::Type getQuantFloatType(Value v);

class ScfTypeConverter : public TypeConverter {
public:
  ScfTypeConverter() {
    // The order of type conversion is important: later ones are tried earlier.
    addConversion([](Type type) { return type; });
    addConversion([&](TensorType type) -> std::optional<Type> {
      if (isLegal(type.getElementType()))
        return type;
      return std::nullopt;
    });

    addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                                 ValueRange inputs,
                                 Location loc) -> std::optional<Value> {
      if (inputs.size() != 1)
        return std::nullopt;

      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });
  }

  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(funcType.getInputs(),
                                                       funcType.getResults()),
                        [this](mlir::Type type) { return isLegal(type); });
  }

  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }
};

class IfOpLowering : public ConversionPatternEx {
public:
  explicit IfOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPatternEx(typeConverter, top::IfOp::getOperationName(), 1, ctx) {}

protected:
  LogicalResult
  matchAndRewriteImpl(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter) const override {
    std::vector<mlir::Type> new_types;
    auto real_mode = module::getMode();
    if (module::isF16Modes()) {
      real_mode = module::Mode::F16;
    } else if (module::isBF16Modes()) {
      real_mode = module::Mode::BF16;
    }
    for (int i = 0; i < op->getNumResults(); i++) {
      switch (real_mode) {
      case module::Mode::INT8:
        new_types.push_back(
            getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
        break;
      case module::Mode::INT4:
        new_types.push_back(
            getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
        break;
      case module::Mode::F16:
        new_types.push_back(
            getQuantFloatType<mlir::Float16Type>(op->getResult(i)));
        break;
      case module::Mode::BF16:
        new_types.push_back(
            getQuantFloatType<mlir::BFloat16Type>(op->getResult(i)));
        break;
      default:
        new_types.emplace_back(op->getResultTypes()[i]);
        break;
      }
    }

    auto tpuIfOp = rewriter.create<tpu::IfOp>(
        op->getLoc(), new_types, op->getOperands(), op->getAttrs());
    rewriter.createBlock(&(tpuIfOp.getThenBranch()));
    rewriter.createBlock(&(tpuIfOp.getElseBranch()));
    auto ifOp = dyn_cast<top::IfOp>(op);
    graphToTpuBranch(rewriter, op->getLoc(), ifOp.getThenBranch(),
                     tpuIfOp.getThenBranch());
    graphToTpuBranch(rewriter, op->getLoc(), ifOp.getElseBranch(),
                     tpuIfOp.getElseBranch());
    op->replaceAllUsesWith(tpuIfOp.getOperation());
    rewriter.eraseOp(op);
    return success();
  }

  bool shouldPrint(Operation *op) const override { return false;}

private:
  void graphToTpuBranch(PatternRewriter &rewriter, Location loc, Region &graph,
                        Region &tpuBranch) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    rewriter.eraseBlock(&tpuBranch.back());
    tpuBranch.takeBody(graph);
    rewriter.setInsertionPointToEnd(&tpuBranch.back());

    Operation *returnOp = tpuBranch.back().getTerminator();
    rewriter.replaceOpWithNewOp<tpu::YieldOp>(returnOp,
                                              returnOp->getOperands());
  }
};


class LoopOpLowering : public ConversionPatternEx {
public:
  explicit LoopOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPatternEx(typeConverter, top::LoopOp::getOperationName(), 1, ctx) {}

protected:
  LogicalResult
  matchAndRewriteImpl(Operation *op, ArrayRef<Value> operands,
                      ConversionPatternRewriter &rewriter) const override {
    std::vector<mlir::Type> new_types;
    auto real_mode = module::getMode();
    if (module::isF16Modes()) {
      real_mode = module::Mode::F16;
    } else if (module::isBF16Modes()) {
      real_mode = module::Mode::BF16;
    }
    for (int i = 0; i < op->getNumResults(); i++) {
      switch (real_mode) {
      case module::Mode::INT8:
        new_types.push_back(
            getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
        break;
      case module::Mode::INT4:
        new_types.push_back(
            getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
        break;
      case module::Mode::F16:
        new_types.push_back(
            getQuantFloatType<mlir::Float16Type>(op->getResult(i)));
        break;
      case module::Mode::BF16:
        new_types.push_back(
            getQuantFloatType<mlir::BFloat16Type>(op->getResult(i)));
        break;
      default:
        new_types.emplace_back(op->getResultTypes()[i]);
        break;
      }
    }

    auto tpuLoopOp = rewriter.create<tpu::LoopOp>(
        op->getLoc(), new_types, op->getOperands(), op->getAttrs());
    rewriter.createBlock(&(tpuLoopOp.getBody()));
    auto loopOp = dyn_cast<top::LoopOp>(op);
    graphToTpuBranch(rewriter, op->getLoc(), loopOp.getBody(),
                     tpuLoopOp.getBody());

    for (int i = 0; i < tpuLoopOp.getBody().getNumArguments(); i++) {
      auto type = tpuLoopOp.getOperand(i).getType();
      tpuLoopOp.getBody().getArgument(i).setType(type);
    }

    auto yieldOp = tpuLoopOp.getBody().front().getTerminator();
    // update the loopop's output
    for (int i = 0; i < tpuLoopOp.v_final().size(); i++) {
      auto type = yieldOp->getOperand(i + 1).getType();
      tpuLoopOp.getResult(i).setType(type);
    }
    op->replaceAllUsesWith(tpuLoopOp.getOperation());
    rewriter.eraseOp(op);
    return success();
  }

  bool shouldPrint(Operation *op) const override { return false;}
private:
  void graphToTpuBranch(PatternRewriter &rewriter, Location loc, Region &graph,
                        Region &tpuBranch) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    rewriter.eraseBlock(&tpuBranch.back());
    tpuBranch.takeBody(graph);
    rewriter.setInsertionPointToEnd(&tpuBranch.back());

    Operation *returnOp = tpuBranch.back().getTerminator();
    rewriter.replaceOpWithNewOp<tpu::YieldOp>(returnOp,
                                              returnOp->getOperands());
  }
};

struct LoweringConfig {
  static bool isQuantized;
  static bool doWinograd;
  static std::map<std::string, module::Mode> quantize_map;
};

static module::Mode getOpQuantMode(Operation *op) {
  auto real_mode = module::getMode();
  auto op_name = module::getName(op);
  auto iter = LoweringConfig::quantize_map.find(op_name.str());
  if (iter != LoweringConfig::quantize_map.end()) {
    real_mode = iter->second;
  }
  if ((module::isCV18xx() || module::isMARS3()) && real_mode == module::Mode::F16) {
    return module::Mode::BF16;
  }
  if (!isa<top::ConvOp, top::MatMulOp>(op) && real_mode == module::Mode::INT4) {
    return module::Mode::INT8;
  }
  return real_mode;
}

template <typename OpTy>
class TopLowering : public OpRewriterPatternEx<OpTy> {
public:
  TopLowering(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context) {}

protected:
  mlir::LogicalResult matchAndRewriteImpl(OpTy opTy,
                                          mlir::PatternRewriter &rewriter) const override {
    Operation *op = opTy.getOperation();

    bool isQuantized = LoweringConfig::isQuantized;
    if (isQuantized) {
      auto stype = module::getStorageType(opTy.getODSResults(0)[0]);
      if (stype.isF32()) {
        if (!isa<top::CastOp,top::Yuv2rgbFormulaOp>(op)) {
          module::removeAttr(op, "round_mode");
          module::removeAttr(op, "first_round_mode");
        }
        LoweringF32(rewriter, opTy);
      } else if (stype.isF16()) {
        if (!isa<top::CastOp,top::Yuv2rgbFormulaOp>(op)) {
          module::removeAttr(op, "round_mode");
          module::removeAttr(op, "first_round_mode");
        }
        LoweringF16(rewriter, opTy);
      } else {
        LoweringQuantized(rewriter, opTy);
      }
      return success();
    }
    auto real_mode = getOpQuantMode(op);
    if (!isa<top::CastOp,top::Yuv2rgbFormulaOp>(op))
    {
      module::removeAttr(op, "round_mode");
      module::removeAttr(op, "first_round_mode");
    }
    switch (real_mode) {
    case module::Mode::INT8:
      if (auto conv = dyn_cast<top::ConvOp>(op)) {
        conv.setDoWinograd(LoweringConfig::doWinograd);
      }
      LoweringINT8(rewriter, opTy, module::isAsymmetric());
      break;
    case module::Mode::INT4:
      LoweringINT4(rewriter, opTy, module::isAsymmetric());
      break;
    case module::Mode::F16:
    case module::Mode::W8F16:
    case module::Mode::W4F16:
      LoweringF16(rewriter, opTy);
      break;
    case module::Mode::BF16:
    case module::Mode::W8BF16:
    case module::Mode::W4BF16:
      LoweringBF16(rewriter, opTy);
      break;
    case module::Mode::F8:
    case module::Mode::F8E4M3:
    case module::Mode::F8E5M2:
      LoweringF8(rewriter, opTy);
      break;
    default:
      LoweringF32(rewriter, opTy);
      break;
    }
    return success();
  }

public:
  bool shouldPrint(OpTy opTy) const override { return false;}
  virtual void LoweringINT8(PatternRewriter &rewriter, OpTy opTy,
                            bool asymmetric) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
  virtual void LoweringINT4(PatternRewriter &rewriter, OpTy opTy,
                            bool asymmetric) const {
    LoweringINT8(rewriter, opTy, asymmetric);
  }
  virtual void LoweringBF16(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
  virtual void LoweringF16(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
  virtual void LoweringF32(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
  virtual void LoweringF8(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
  virtual void LoweringQuantized(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }
};

template <typename OpTy>
class TopShapeLowering : public OpRewriterPatternEx<OpTy> {
public:
  TopShapeLowering(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context) {}

protected:
  mlir::LogicalResult matchAndRewriteImpl(OpTy opTy,
                                          mlir::PatternRewriter &rewriter) const override {
    Lowering(rewriter, opTy);
    return success();
  }

public:
  virtual void Lowering(PatternRewriter &rewriter, OpTy opTy) const {
    UNREACHABLE_OP("Not Implemented", opTy);
  }

  bool shouldPrint(OpTy opTy) const override { return false;}
};

// Lowering to a new Operation, with the same operands and same attrs, and
// newType
template <typename OpTy>
static OpTy lowering_common(PatternRewriter &rewriter, Operation *from,
                            Type newType, int num_operands = 0) {
  auto stype = module::getStorageType(newType);
  std::vector<Value> operands;
  int in_num_ops = from->getNumOperands();
  if (num_operands > 0) {
    in_num_ops = std::min(num_operands, in_num_ops);
  }
  for (int i = 0; i < in_num_ops; ++i) {
    auto in = from->getOperand(i);
    if (module::isWeight(in)) {
      [[maybe_unused]] auto wOp = in.getDefiningOp<top::WeightOp>();
      [[maybe_unused]] auto wtype = module::getStorageType(in);
      if (stype.isF16()) {
        operands.push_back(wOp.clone_f16(from));
      } else if (stype.isBF16()) {
        operands.push_back(wOp.clone_bf16(from));
      } else if (stype.isFloat8E5M2()) {
        operands.push_back(wOp.clone_f8e5m2(from));
      } else if (stype.isFloat8E4M3FN()) {
        operands.push_back(wOp.clone_f8e4m3(from, false));
      } else {
        operands.push_back(in);
      }
    } else {
      operands.push_back(in);
    }
  }
  if (num_operands > from->getNumOperands()) {
    auto noneOp = module::getNoneOp(from);
    for (int i = from->getNumOperands(); i < num_operands; i++) {
      operands.push_back(noneOp);
    }
  }
  return rewriter.replaceOpWithNewOp<OpTy>(from, newType, operands,
                                           from->getAttrs());
}

// lowering to a new Operation, with same operands and same attrs, and quantize
// f32 output to int8 output
template <typename OpTy>
static OpTy lowering_common_int8(PatternRewriter &rewriter, Operation *from,
                                 bool asymmetric = false,
                                 int num_operands = 0) {
  assert(from->getNumResults() == 1);
  Type newType;
  if (module::isUniformQuantized(from->getResult(0)))
    newType = from->getResult(0).getType();
  else
    newType = getQuantInt8Type(from->getResult(0), asymmetric);
  return lowering_common<OpTy>(rewriter, from, newType, num_operands);
}

Type getQuantF8E4M3Type(Value v);
Type getQuantF8E5M2Type(Value v);

template <typename OpTy>
static OpTy lowering_common_f8(PatternRewriter &rewriter, Operation *from,
                               bool isE4, int num_operands = 0) {
  assert(from->getNumResults() == 1);
  if (isE4) {
    auto newType = getQuantF8E4M3Type(from->getResult(0));
    return lowering_common<OpTy>(rewriter, from, newType, num_operands);
  } else {
    auto newType = getQuantF8E5M2Type(from->getResult(0));
    return lowering_common<OpTy>(rewriter, from, newType, num_operands);
  }
}

template <typename ElemTy = Float32Type>
static mlir::Type getQuantFloatType(Value v) {
  Type newType = v.getType();
  if (newType.isa<mlir::NoneType>()) {
    return newType;
  }
  auto sType = module::getStorageType(v);
  if (sType.isa<ElemTy>() == false) {
    auto shape = module::getShape(v);
    auto ctx = v.getContext();
    if (module::isCalibratedType(v)) {
      auto caliType = module::getCalibratedType(v);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          ElemTy::get(ctx), caliType.getMin(), caliType.getMax());
      newType = RankedTensorType::get(shape, newCaliType);
    } else {
      newType = RankedTensorType::get(shape, ElemTy::get(ctx));
    }
  }
  return newType;
}

static mlir::Type getQuantBF16Type(Value v) {
  return getQuantFloatType<BFloat16Type>(v);
}

static mlir::Type getQuantF16Type(Value v) {
  return getQuantFloatType<Float16Type>(v);
}

// lowering to f32/f16/bf16
template <typename OpTy, typename ElemTy>
static OpTy lowering_common_float(PatternRewriter &rewriter, Operation *from,
                                  int num_operands = 0) {
  assert(from->getNumResults() == 1);
  auto newType = getQuantFloatType<ElemTy>(from->getResult(0));
  return lowering_common<OpTy>(rewriter, from, newType, num_operands);
}

template <typename OpTy>
static OpTy lowering_common_f32(PatternRewriter &rewriter, Operation *from,
                                int num_operands = 0) {
  return lowering_common_float<OpTy, Float32Type>(rewriter, from, num_operands);
}

template <typename OpTy>
static OpTy lowering_common_bf16(PatternRewriter &rewriter, Operation *from,
                                 int num_operands = 0) {
  return lowering_common_float<OpTy, BFloat16Type>(rewriter, from,
                                                   num_operands);
}

template <typename OpTy>
static OpTy lowering_common_f16(PatternRewriter &rewriter, Operation *from,
                                int num_operands = 0) {
  return lowering_common_float<OpTy, Float16Type>(rewriter, from, num_operands);
}

// from int8 to int8, convert one (scale zp) to another (scale zp)
Value do_transfer(Value in, Value out, bool asymmetric);
Value do_transfer_fp(Value in, Value out, bool asymmetric,
                     tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero);

// from int8 to int32
Value do_dequant(Location name_loc, Value input, Type to_type,
                 int64_t multiplier, int64_t rshift, tpu::DequantMode mode,
                 int64_t lshift,
                 tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero);

// from int32 to int8
Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode,
                 tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero);

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode, tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero);

Value do_requant_axis(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode, tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero, int64_t rq_axis=1, bool fuse_rq = false);

Value do_requantFp(
    Value input, double scale, double offset, Type to_type,
    std::string &to_name,
    tpu::RequantMode mode = tpu::RequantMode::MultiplierShift,
    tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero,
    tpu::RoundMode first_rmode = tpu::RoundMode::HalfAwayFromZero);

Value do_requantFp(
    Value input, Value quant, Type to_type, bool tensorType,
    std::string &to_name, tpu::RequantMode mode,
    tpu::RoundMode rmode = tpu::RoundMode::HalfAwayFromZero,
    tpu::RoundMode first_rmode = tpu::RoundMode::HalfAwayFromZero);

tpu::RequantMode get_requant_mode(std::string mode);
tpu::DequantMode get_dequant_mode(std::string mode);
tpu::RoundMode get_round_mode(std::string mode);

template <typename OpTy>
Value do_binary_saclar(Value input, Type to_type, int64_t scalar,
                       const char *suffix = "_binary") {
  [[maybe_unused]] auto from_stype = module::getStorageType(input);
  [[maybe_unused]] auto to_stype = module::getStorageType(to_type);
  auto ctx = input.getContext();
  OpBuilder builder(ctx);
  auto newType = to_type;
  newType = RankedTensorType::get(module::getShape(input), to_stype);

  builder.setInsertionPointAfterValue(input);
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(scalar)));

  std::string new_name = module::getName(input.getDefiningOp()).str() + suffix;
  auto name_loc = NameLoc::get(builder.getStringAttr(new_name));
  auto newOp =
      builder.create<OpTy>(name_loc, newType, ValueRange{input}, attrs);
  return newOp.getOutput();
}

Value do_f8_relu(Value input, Type to_type, double relu_limit);
Value do_reshape(Value input, RankedTensorType to_type);
Value do_transpose(Location name_loc, Value input, std::vector<int64_t> &order);
Value do_weight_dequant(Value input, Type to_type, int64_t multiplier,
                        int64_t shift, int64_t lshift);
int32_t do_const_dequant(Value input, int64_t multiplier, int64_t shift,
                         int64_t lshift);

// try to insert tpu.Host2DeviceOp at input #idx
void try_insert_host2device(Operation *op, uint32_t idx);
// try to insert tpu.Device2HostOp at input #idx
void try_insert_device2host(Operation *op, uint32_t idx);

Value insert_device2host(Value v, Type to, Operation *user = nullptr);

bool isa_shape_subnet_op(Operation *op);
} // namespace tpu_mlir
