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

using namespace llvm;

namespace tpu_mlir {
// ================================
// Lowering Helper Apis
// ================================

mlir::Type getQuantInt8Type(Value v, bool asymmetric = false);
mlir::Type getQuantIntType(Value v, double scale, double offset, int bits = 8);
mlir::Type getQuantInt4Type(Value v, bool asymmetric = false);
mlir::Type getQuantBoolType(Value v);

template <typename ElemTy = Float32Type>
static mlir::Type getQuantFloatType(Value v);

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

class IfOpLowering : public ConversionPattern {
public:
  explicit IfOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, top::IfOp::getOperationName(), 1,
                          ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    std::vector<mlir::Type> new_types;
    auto real_mode = module::getMode();
    for (int i = 0; i < op->getNumResults(); i++)
    {
      switch (real_mode) {
        case module::Mode::INT8:
          new_types.push_back(getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
          break;
        case module::Mode::INT4:
          new_types.push_back(getQuantInt8Type(op->getResult(i), module::isAsymmetric()));
          break;
        case module::Mode::F16:
          new_types.push_back(getQuantFloatType<mlir::Float16Type>(op->getResult(i)));
          break;
        case module::Mode::BF16:
          new_types.push_back(getQuantFloatType<mlir::BFloat16Type>(op->getResult(i)));
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
  static std::map<std::string, module::Mode> quantize_map;
};

template <typename OpTy> class TopLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Operation *op = opTy.getOperation();
    bool isQuantized = LoweringConfig::isQuantized;
    if (isQuantized) {
      LoweringQuantized(rewriter, opTy);
      return success();
    }
    auto real_mode = module::getMode();
    auto op_name = module::getName(op);
    auto iter = LoweringConfig::quantize_map.find(op_name.str());
    if (iter != LoweringConfig::quantize_map.end()) {
      real_mode = iter->second;
    }
    switch (real_mode) {
    case module::Mode::INT8:
      LoweringINT8(rewriter, opTy, module::isAsymmetric());
      break;
    case module::Mode::INT4:
      if (isa<top::ConvOp, top::MatMulOp>(op)) {
        LoweringINT4(rewriter, opTy, module::isAsymmetric());
      } else {
        LoweringINT8(rewriter, opTy, module::isAsymmetric());
      }
      break;
    case module::Mode::F16:
      if (module::isCV18xx()) {
        LoweringBF16(rewriter, opTy);
      } else {
        LoweringF16(rewriter, opTy);
      }
      break;
    case module::Mode::BF16:
      LoweringBF16(rewriter, opTy);
      break;
    default:
      LoweringF32(rewriter, opTy);
      break;
    }
    return success();
  }

public:
  virtual void LoweringINT8(PatternRewriter &rewriter, OpTy opTy,
                            bool asymmetric) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringINT4(PatternRewriter &rewriter, OpTy opTy,
                            bool asymmetric) const {
    LoweringINT8(rewriter, opTy, asymmetric);
  }
  virtual void LoweringBF16(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringF16(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringF32(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
  virtual void LoweringQuantized(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
};

template <typename OpTy>
class TopShapeLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy opTy,
                                PatternRewriter &rewriter) const override {
    Lowering(rewriter, opTy);
    return success();
  }

public:
  virtual void Lowering(PatternRewriter &rewriter, OpTy opTy) const {
    llvm_unreachable("Not Implemented");
  }
};

// Lowering to a new Operation, with the same operands and same attrs, and
// newType
template <typename OpTy>
static void lowering_common(PatternRewriter &rewriter, Operation *from,
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
      [[maybe_unused]]auto wOp = in.getDefiningOp<top::WeightOp>();
      [[maybe_unused]]auto wtype = module::getStorageType(in);
      if (stype.isF16()) {
        operands.push_back(wOp.clone_f16(from));
      } else if (stype.isBF16()) {
        operands.push_back(wOp.clone_bf16(from));
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
  rewriter.replaceOpWithNewOp<OpTy>(from, newType, operands, from->getAttrs());
}

// lowering to a new Operation, with same operands and same attrs, and quantize
// f32 output to int8 output
template <typename OpTy>
static void lowering_common_int8(PatternRewriter &rewriter, Operation *from,
                                 bool asymmetric = false,
                                 int num_operands = 0) {
  assert(from->getNumResults() == 1);
  auto newType = getQuantInt8Type(from->getResult(0), asymmetric);
  lowering_common<OpTy>(rewriter, from, newType, num_operands);
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
static void lowering_common_float(PatternRewriter &rewriter, Operation *from,
                                  int num_operands = 0) {
  assert(from->getNumResults() == 1);
  auto newType = getQuantFloatType<ElemTy>(from->getResult(0));
  lowering_common<OpTy>(rewriter, from, newType, num_operands);
}

template <typename OpTy>
static void lowering_common_f32(PatternRewriter &rewriter, Operation *from,
                                int num_operands = 0) {
  lowering_common_float<OpTy, Float32Type>(rewriter, from, num_operands);
}

template <typename OpTy>
static void lowering_common_bf16(PatternRewriter &rewriter, Operation *from,
                                 int num_operands = 0) {
  lowering_common_float<OpTy, BFloat16Type>(rewriter, from, num_operands);
}

template <typename OpTy>
static void lowering_common_f16(PatternRewriter &rewriter, Operation *from,
                                int num_operands = 0) {
  lowering_common_float<OpTy, Float16Type>(rewriter, from, num_operands);
}

// from int8 to int8, convert one (scale zp) to another (scale zp)
Value do_transfer(Value in, Value out, bool asymmetric);
Value do_transfer_fp(Value in, Value out, bool asymmetric);

// from int8 to int32
Value do_dequant(Location name_loc, Value input, Type to_type,
                 int64_t multiplier, int64_t rshift, tpu::DequantMode mode,
                 int64_t lshift);

// from int8 to int32
Value do_requant(Location name_loc, Value input, Type to_type, bool tensorType,
                 int64_t multiplier, int64_t shift, tpu::RequantMode mode);

Value do_requant(Location name_loc, Value input, Value quant, Type to_type,
                 bool tensorType, tpu::RequantMode mode);

Value do_requantFp(Value input, double scale, double offset, Type to_type,
                   std::string &to_name,
                   tpu::RequantMode mode = tpu::RequantMode::MultiplierShift);

template <typename OpTy>
Value do_binary_saclar(Value input, Type to_type, int64_t scalar,
                       const char *suffix = "_binary") {
  [[maybe_unused]]auto from_stype = module::getStorageType(input);
  [[maybe_unused]]auto to_stype = module::getStorageType(to_type);
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

} // namespace tpu_mlir
